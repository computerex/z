package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ── Message types ───────────────────────────────────────────────────────

// Message represents a conversation message.
type Message struct {
	Role           string      `json:"role"`
	Content        interface{} `json:"content"` // string or []ContentBlock
	ProviderBlocks interface{} `json:"provider_blocks,omitempty"`
}

// ContentBlock for multimodal messages.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// MessageContent returns the text content of a message.
func MessageContent(m Message) string {
	switch v := m.Content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, block := range v {
			if bm, ok := block.(map[string]interface{}); ok {
				if t, ok := bm["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// ── Session struct ──────────────────────────────────────────────────────

// Session holds the conversation state.
type Session struct {
	Name          string    `json:"name,omitempty"`
	Workspace     string    `json:"workspace"`
	Messages      []Message `json:"messages"`
	Context       []CtxItem `json:"context,omitempty"`
	ContextNextID int       `json:"context_next_id"`
	Todos         interface{} `json:"todos,omitempty"`
	CreatedAt     string    `json:"created_at,omitempty"`
	UpdatedAt     string    `json:"updated_at,omitempty"`

	needsSystemPrompt bool `json:"-"` // transient: true if system prompt needs (re)generation
}

// CtxItem is a stored context item (file read, tool result, etc.).
type CtxItem struct {
	ID        int       `json:"id"`
	Type      string    `json:"type"`
	Source    string    `json:"source"`
	Content   string    `json:"content"`
	AddedAt   float64   `json:"added_at"`
	LineRange []int     `json:"line_range,omitempty"`
}

// NewSession creates an empty session.
func NewSession(workspace string) *Session {
	now := time.Now().Format(time.RFC3339)
	return &Session{
		Workspace:     workspace,
		Messages:      make([]Message, 0),
		Context:       make([]CtxItem, 0),
		ContextNextID: 1,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
}

// ── Save / Load ─────────────────────────────────────────────────────────

// SaveSession writes the session to disk.
func SaveSession(sess *Session, path string) error {
	sess.UpdatedAt = time.Now().Format(time.RFC3339)

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("save session: create dir: %w", err)
	}

	data, err := json.MarshalIndent(sess, "", "  ")
	if err != nil {
		return fmt.Errorf("save session: marshal: %w", err)
	}

	// Atomic write: write to temp then rename
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return fmt.Errorf("save session: write: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		// Fallback: direct write (rename may fail cross-device)
		os.Remove(tmp)
		return os.WriteFile(path, data, 0644)
	}
	return nil
}

// LoadSession reads a session from disk.
func LoadSession(path string) (*Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load session: %w", err)
	}

	var sess Session
	if err := json.Unmarshal(data, &sess); err != nil {
		return nil, fmt.Errorf("load session: parse: %w", err)
	}

	// Integrity: ensure messages[0] is a valid system prompt
	if len(sess.Messages) > 0 {
		first := sess.Messages[0]
		if first.Role != "system" {
			logWarn("Session missing system prompt, marking for regeneration")
			sess.needsSystemPrompt = true
		} else {
			content := MessageContent(first)
			if !strings.Contains(content, "TOOL USE") || !strings.Contains(content, "read_file") {
				logWarn("Session system prompt appears corrupted, marking for replacement")
				sess.Messages = sess.Messages[1:]
				sess.needsSystemPrompt = true
			}
		}
	} else {
		sess.needsSystemPrompt = true
	}

	return &sess, nil
}

// ── Session helpers ─────────────────────────────────────────────────────

// AppendMessage adds a message to the session.
func (s *Session) AppendMessage(role, content string) {
	s.Messages = append(s.Messages, Message{Role: role, Content: content})
}

// AppendAssistant adds an assistant message with optional provider blocks.
func (s *Session) AppendAssistant(content string, providerBlocks interface{}) {
	s.Messages = append(s.Messages, Message{
		Role:           "assistant",
		Content:        content,
		ProviderBlocks: providerBlocks,
	})
}

// LastUserMessage returns the content of the most recent user message.
func (s *Session) LastUserMessage() string {
	for i := len(s.Messages) - 1; i >= 0; i-- {
		if s.Messages[i].Role == "user" {
			return MessageContent(s.Messages[i])
		}
	}
	return ""
}

// AddContext stores a context item (file read, tool result, etc.).
func (s *Session) AddContext(itemType, source, content string, lineRange []int) int {
	id := s.ContextNextID
	s.ContextNextID++
	item := CtxItem{
		ID:        id,
		Type:      itemType,
		Source:    source,
		Content:   content,
		AddedAt:   float64(time.Now().Unix()),
		LineRange: lineRange,
	}
	s.Context = append(s.Context, item)
	return id
}

// EstimateTokens gives a rough token estimate for the conversation.
func (s *Session) EstimateTokens() int {
	total := 0
	for _, m := range s.Messages {
		content := MessageContent(m)
		// Rough estimate: 1 token per 4 chars
		total += len(content) / 4
	}
	return total
}

// GenerateSessionName creates a name from the first user message.
func GenerateSessionName(firstMessage string) string {
	// Take first 40 chars of user message, sanitize for filename
	name := firstMessage
	if len(name) > 40 {
		name = name[:40]
	}
	name = strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == ' ' {
			return r
		}
		return '_'
	}, name)
	name = strings.TrimSpace(name)
	if name == "" {
		name = fmt.Sprintf("session_%d", time.Now().Unix())
	}
	return name
}
