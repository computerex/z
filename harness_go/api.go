package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// StreamingResponse represents a streaming response from the API
type StreamingResponse struct {
	Content          string
	Usage            map[string]int
	Interrupted      bool
	FinishReason     string
	WebSearch        bool
	WebSearchResults []WebSearchResult
}

// WebSearchResult represents a web search result
type WebSearchResult struct {
	Title   string
	Link    string
	Media   string
	Date    string
	Content string
}

// APIClient handles API calls
type APIClient struct {
	APIKey      string
	BaseURL     string
	Model       string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
}

// NewAPIClient creates a new API client
func NewAPIClient(apiKey, baseURL, model string) *APIClient {
	if !strings.HasSuffix(baseURL, "/") {
		baseURL = baseURL + "/"
	}
	return &APIClient{
		APIKey:      apiKey,
		BaseURL:     baseURL,
		Model:       model,
		Temperature: 0.7,
		MaxTokens:   8192,
		Timeout:     120 * time.Second,
	}
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    []Message       `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       json.RawMessage `json:"tools,omitempty"`
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a choice in the response
type Choice struct {
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
	Message      Message `json:"message"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Chat sends a non-streaming chat request
func (c *APIClient) Chat(messages []Message) (*ChatResponse, error) {
	url := c.BaseURL + "chat/completions"

	reqBody := ChatRequest{
		Model:       c.Model,
		Messages:    messages,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
		Stream:      false,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	client := &http.Client{Timeout: c.Timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &chatResp, nil
}

// ChatStream sends a streaming chat request
func (c *APIClient) ChatStream(messages []Message, onChunk func(string), onInterrupt func() bool) (*StreamingResponse, error) {
	url := c.BaseURL + "chat/completions"

	reqBody := ChatRequest{
		Model:       c.Model,
		Messages:    messages,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
		Stream:      true,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	client := &http.Client{Timeout: c.Timeout * 2}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var fullContent strings.Builder
	reader := bufio.NewReader(resp.Body)

	for {
		if onInterrupt != nil && onInterrupt() {
			return &StreamingResponse{
				Content:     fullContent.String(),
				Interrupted: true,
			}, nil
		}

		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}

		line = strings.TrimSpace(line)

		// Skip empty lines and SSE comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Remove "data: " prefix if present
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}

		line = strings.TrimSpace(line)
		if line == "" || line == "[DONE]" {
			continue
		}

		// Extract content from JSON chunk
		content := extractContentFromChunk(line)
		if content != "" {
			fullContent.WriteString(content)
			onChunk(content)
		}
	}

	return &StreamingResponse{
		Content:      fullContent.String(),
		FinishReason: "stop",
	}, nil
}

// extractContentFromChunk extracts the content field from a streaming JSON chunk
func extractContentFromChunk(chunk string) string {
	// Find "content": "..." pattern that correctly handles escaped quotes
	// The pattern matches: "content": " followed by any chars (including escaped \") until unescaped "
	contentPattern := regexp.MustCompile(`"content"\s*:\s*"((?:[^"\\]|\\.)*)"`)
	match := contentPattern.FindStringSubmatch(chunk)
	if match != nil {
		content := match[1]
		// Decode JSON escape sequences using the json package
		var decoded string
		if err := json.Unmarshal([]byte(`"` + content + `"`), &decoded); err == nil {
			return decoded
		}
		// Fallback: return raw content if decoding fails
		return content
	}
	return ""
}

// VisionRequest represents a vision API request
type VisionRequest struct {
	Model     string         `json:"model"`
	Messages  []VisionMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`
}

// VisionMessage represents a message with image
type VisionMessage struct {
	Role    string         `json:"role"`
	Content []VisionContent `json:"content"`
}

// VisionContent represents content with image or text
type VisionContent struct {
	Type     string   `json:"type"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
	Text     string   `json:"text,omitempty"`
}

// ImageURL represents an image URL
type ImageURL struct {
	URL string `json:"url"`
}

// AnalyzeImage analyzes an image using vision API
func (c *APIClient) AnalyzeImage(imagePath, question string) (string, error) {
	content, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image: %w", err)
	}

	// Determine mime type
	mimeType := "image/png"
	ext := strings.ToLower(filepath.Ext(imagePath))
	switch ext {
	case ".jpg", ".jpeg":
		mimeType = "image/jpeg"
	case ".png":
		mimeType = "image/png"
	case ".gif":
		mimeType = "image/gif"
	case ".webp":
		mimeType = "image/webp"
	}

	// Encode to base64
	base64Img := base64.StdEncoding.EncodeToString(content)

	// Use coding endpoint for vision
	url := strings.Replace(c.BaseURL, "/api/coding/paas/v4/", "/api/coding/paas/v4/chat/completions", 1)

	messages := []VisionMessage{
		{
			Role: "user",
			Content: []VisionContent{
				{
					Type: "image_url",
					ImageURL: &ImageURL{
						URL: fmt.Sprintf("data:%s;base64,%s", mimeType, base64Img),
					},
				},
				{
					Type: "text",
					Text: question,
				},
			},
		},
	}

	reqBody := VisionRequest{
		Model:     "glm-4.6v",
		Messages:  messages,
		MaxTokens: 2048,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if len(chatResp.Choices) > 0 {
		return chatResp.Choices[0].Message.Content, nil
	}

	return "", fmt.Errorf("empty response from vision API")
}