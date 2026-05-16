package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// StreamChunk is delivered to the callback during streaming.
type StreamChunk struct {
	Content      string
	Reasoning    string
	FinishReason string
}

// StreamingClient handles API streaming to various providers.
type StreamingClient struct {
	APIBase         string
	APIKey          string
	Model           string
	Temperature     float64
	MaxTokens       int
	ReasoningEffort string
	httpClient      *http.Client

	// Copilot-specific
	isCopilot       bool
	copilotOpaque   string

	// Bedrock-specific
	isBedrock       bool
}

// NewStreamingClient creates a client based on the config.
func NewStreamingClient(cfg *Config) *StreamingClient {
	sc := &StreamingClient{
		APIBase:         cfg.APIBase,
		APIKey:          cfg.APIKey,
		Model:           normalizeModelForAPI(cfg.Model, cfg.APIBase),
		Temperature:     cfg.Temperature,
		MaxTokens:       cfg.MaxTokens,
		ReasoningEffort: cfg.ReasoningEffort,
		httpClient: &http.Client{
			// No overall timeout — streaming bodies can take minutes.
			// Connection-level timeouts prevent blocking on dead servers.
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				TLSHandshakeTimeout:   15 * time.Second,
				ResponseHeaderTimeout: 60 * time.Second,
			},
		},
	}

	if strings.HasPrefix(cfg.APIKey, "oauth:") &&
		(strings.Contains(strings.ToLower(cfg.APIBase), "copilot") ||
			strings.Contains(strings.ToLower(cfg.APIBase), "githubcopilot")) {
		sc.isCopilot = true
	}

	// Bedrock uses a completely different API (AWS Converse) — not OpenAI SSE
	if strings.Contains(strings.ToLower(cfg.APIBase), "bedrock") &&
		strings.Contains(strings.ToLower(cfg.APIBase), "amazonaws.com") {
		sc.isBedrock = true
	}

	return sc
}

// ChatStream sends messages to the API and calls onChunk for each streaming event.
func (sc *StreamingClient) ChatStream(messages []map[string]interface{}, onChunk func(StreamChunk)) error {
	if sc.isCopilot {
		return sc.chatStreamCopilot(messages, onChunk)
	}
	if sc.isBedrock {
		return sc.chatStreamBedrock(messages, onChunk)
	}
	return sc.chatStreamSSE(messages, onChunk)
}

func (sc *StreamingClient) chatStreamCopilot(messages []map[string]interface{}, onChunk func(StreamChunk)) error {
	// Copilot uses the OAuth access_token directly as Bearer token
	// (no v2/token exchange needed)
	accessToken := strings.TrimPrefix(sc.APIKey, "oauth:")

	body := map[string]interface{}{
		"model":    sc.Model,
		"messages": sc.injectOpaque(messages),
		"stream":   true,
	}
	if sc.ReasoningEffort != "" && sc.ReasoningEffort != "none" {
		body["reasoning_effort"] = sc.ReasoningEffort
	}
	if sc.Temperature > 0 {
		body["temperature"] = sc.Temperature
	}
	if sc.MaxTokens > 0 {
		body["max_tokens"] = sc.MaxTokens
	}

	bodyJSON, _ := json.Marshal(body)

	url := strings.TrimRight(sc.APIBase, "/") + "/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyJSON))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "harness/1.0.0")
	req.Header.Set("x-initiator", "agent")
	req.Header.Set("Openai-Intent", "conversation-edits")

	return sc.doSSEStream(req, onChunk)
}

func (sc *StreamingClient) chatStreamSSE(messages []map[string]interface{}, onChunk func(StreamChunk)) error {
	body := map[string]interface{}{
		"model":    sc.Model,
		"messages": messages,
		"stream":   true,
		"stream_options": map[string]interface{}{
			"include_usage": true,
		},
	}
	if sc.ReasoningEffort != "" && sc.ReasoningEffort != "none" {
		body["reasoning_effort"] = sc.ReasoningEffort
	}
	if sc.Temperature > 0 {
		body["temperature"] = sc.Temperature
	}
	if sc.MaxTokens > 0 {
		body["max_tokens"] = sc.MaxTokens
	}

	bodyJSON, _ := json.Marshal(body)

	url := strings.TrimRight(sc.APIBase, "/") + "/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyJSON))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	if strings.HasPrefix(sc.APIKey, "oauth:") {
		req.Header.Set("Authorization", "Bearer "+strings.TrimPrefix(sc.APIKey, "oauth:"))
	} else {
		req.Header.Set("Authorization", "Bearer "+sc.APIKey)
	}

	return sc.doSSEStream(req, onChunk)
}

func (sc *StreamingClient) doSSEStream(req *http.Request, onChunk func(StreamChunk)) error {
	resp, err := sc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	firstDeltaLogged := false
	reasoningChunks := 0
	contentChunks := 0

	for scanner.Scan() {
		if isInterrupted() {
			onChunk(StreamChunk{FinishReason: "interrupted"})
			break
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event sseEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		// Log raw delta keys on first event to detect unknown reasoning fields
		if !firstDeltaLogged {
			var rawEvent map[string]interface{}
			if json.Unmarshal([]byte(data), &rawEvent) == nil {
				if choices, ok := rawEvent["choices"].([]interface{}); ok && len(choices) > 0 {
					if ch, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := ch["delta"].(map[string]interface{}); ok {
							keys := make([]string, 0, len(delta))
							for k := range delta {
								keys = append(keys, k)
							}
							logDebugf("SSE raw delta keys: %v", keys)
						}
					}
				}
			}
		}

		if len(event.Choices) == 0 {
			continue
		}

		choice := event.Choices[0]
		delta := choice.Delta

		// Log first delta keys for debugging reasoning token presence
		if !firstDeltaLogged {
			firstDeltaLogged = true
			var keys []string
			if delta.Content != "" { keys = append(keys, "content") }
			if delta.ReasoningContent != "" { keys = append(keys, "reasoning_content") }
			if delta.Reasoning != "" { keys = append(keys, "reasoning") }
			if delta.ReasoningText != "" { keys = append(keys, "reasoning_text") }
			if delta.Thinking != "" { keys = append(keys, "thinking") }
			if delta.ReasoningOpaque != "" { keys = append(keys, "reasoning_opaque") }
			if delta.Role != "" { keys = append(keys, "role="+delta.Role) }
			logDebugf("SSE first delta keys: %v", keys)
		}

		chunk := StreamChunk{}

		if delta.Content != "" {
			chunk.Content = delta.Content
			contentChunks++
		}

		// 5-point reasoning field extraction (matches Python _extract_reasoning_from_delta)
		reasoning := delta.ReasoningContent
		if reasoning == "" {
			reasoning = delta.Reasoning
		}
		if reasoning == "" {
			reasoning = delta.ReasoningText
		}
		if reasoning == "" {
			reasoning = delta.Thinking
		}
		// Check provider_specific_fields
		if reasoning == "" && delta.ProviderSpecificFields != nil {
			for _, key := range []string{"thinking", "reasoning", "reasoning_content"} {
				if v, ok := delta.ProviderSpecificFields[key]; ok {
					if s, ok := v.(string); ok && s != "" {
						reasoning = s
						break
					}
				}
			}
		}
		// Check reasoning_details (MiniMax/ZAI format)
		if reasoning == "" && len(delta.ReasoningDetails) > 0 {
			reasoning = extractReasoningDetailsText(delta.ReasoningDetails)
		}
		if reasoning != "" {
			chunk.Reasoning = reasoning
			reasoningChunks++
		}

		if delta.ReasoningOpaque != "" {
			sc.copilotOpaque = delta.ReasoningOpaque
		}
		if choice.Message.ReasoningOpaque != "" {
			sc.copilotOpaque = choice.Message.ReasoningOpaque
		}

		if choice.FinishReason != "" {
			chunk.FinishReason = choice.FinishReason
		}

		if chunk.Content != "" || chunk.Reasoning != "" || chunk.FinishReason != "" {
			onChunk(chunk)
		}
	}

	logDebugf("SSE stream done: reasoning_chunks=%d content_chunks=%d opaque=%v", reasoningChunks, contentChunks, sc.copilotOpaque != "")
	return scanner.Err()
}

func (sc *StreamingClient) injectOpaque(messages []map[string]interface{}) []interface{} {
	out := make([]interface{}, 0, len(messages))
	lastAssistantIdx := -1
	for i, m := range messages {
		if m["role"] == "assistant" {
			lastAssistantIdx = i
		}
	}

	for i, m := range messages {
		msg := make(map[string]interface{})
		for k, v := range m {
			msg[k] = v
		}
		if sc.isCopilot && m["role"] == "assistant" && i == lastAssistantIdx && sc.copilotOpaque != "" {
			msg["reasoning_opaque"] = sc.copilotOpaque
			logDebugf("injectOpaque: attached opaque to assistant msg at index %d/%d (opaque_len=%d)", i, len(messages), len(sc.copilotOpaque))
		}
		out = append(out, msg)
	}
	return out
}

// SSE event structures
type sseEvent struct {
	Choices []sseChoice            `json:"choices"`
	Usage   sseUsage               `json:"usage"`
}

type sseChoice struct {
	Delta        sseDelta   `json:"delta"`
	Message      sseMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

type sseDelta struct {
	Content                string                 `json:"content"`
	ReasoningContent       string                 `json:"reasoning_content"`
	ReasoningText          string                 `json:"reasoning_text"`
	Reasoning              string                 `json:"reasoning"`
	Thinking               string                 `json:"thinking"`
	ReasoningOpaque        string                 `json:"reasoning_opaque"`
	Role                   string                 `json:"role"`
	ProviderSpecificFields map[string]interface{}  `json:"provider_specific_fields"`
	ReasoningDetails       json.RawMessage         `json:"reasoning_details"`
}

type sseMessage struct {
	ReasoningOpaque string `json:"reasoning_opaque"`
}

type sseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// extractReasoningDetailsText flattens MiniMax/OpenAI-compatible reasoning_details
// blocks into plain text. Handles both object and array formats.
func extractReasoningDetailsText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	// Try as string first
	var s string
	if json.Unmarshal(raw, &s) == nil && s != "" {
		return s
	}

	// Try as object with "text" field
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) == nil {
		if textRaw, ok := obj["text"]; ok {
			var text string
			if json.Unmarshal(textRaw, &text) == nil && text != "" {
				return text
			}
		}
		// Try nested reasoning_details
		if nested, ok := obj["reasoning_details"]; ok {
			return extractReasoningDetailsText(nested)
		}
		// Try summary, content, items
		var parts strings.Builder
		for _, key := range []string{"summary", "content", "items"} {
			if sub, ok := obj[key]; ok {
				t := extractReasoningDetailsText(sub)
				if t != "" {
					parts.WriteString(t)
				}
			}
		}
		return parts.String()
	}

	// Try as array
	var arr []json.RawMessage
	if json.Unmarshal(raw, &arr) == nil {
		var parts strings.Builder
		for _, item := range arr {
			t := extractReasoningDetailsText(item)
			if t != "" {
				parts.WriteString(t)
			}
		}
		return parts.String()
	}

	return ""
}

// normalizeModelForAPI strips LiteLLM-style provider prefixes from model names.
// When sending directly to provider APIs, we don't need "openrouter/", "bedrock/", etc.
func normalizeModelForAPI(model, apiBase string) string {
	// Known LiteLLM provider prefixes that should be stripped
	prefixes := []string{
		"openrouter/",
		"bedrock/",
		"together_ai/",
		"ollama/",
		"groq/",
		"minimax/",
		"deepseek/",
	}

	lower := strings.ToLower(model)
	for _, prefix := range prefixes {
		if strings.HasPrefix(lower, prefix) {
			return model[len(prefix):]
		}
	}

	return model
}
