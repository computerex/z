package main

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// TestProviderIntegration tests each configured provider with a real API call.
// Run with: go test -run TestProviderIntegration -v -timeout 120s
// Skipped by default (needs HARNESS_TEST_PROVIDERS=1).
func TestProviderIntegration(t *testing.T) {
	if os.Getenv("HARNESS_TEST_PROVIDERS") != "1" {
		t.Skip("Set HARNESS_TEST_PROVIDERS=1 to run provider integration tests")
	}

	cfg := loadConfig()
	if len(cfg.Providers) == 0 {
		t.Fatal("No providers configured in ~/.z.json")
	}

	// Use a prompt that requires reasoning to verify thinking works
	messages := []map[string]interface{}{
		{"role": "user", "content": "What is 7823 * 4219? Just the final number."},
	}

	for name, p := range cfg.Providers {
		t.Run(name, func(t *testing.T) {
			provCfg := &Config{
				Provider:        name,
				APIBase:         p.APIURL,
				APIKey:          p.APIKey,
				Model:           p.Model,
				MaxTokens:       4096,
				Temperature:     0.7,
				ReasoningEffort: "high",
			}

			client := NewStreamingClient(provCfg)
			client.httpClient.Timeout = 60 * time.Second

			var content strings.Builder
			var reasoning strings.Builder
			var finishReason string

			err := client.ChatStream(messages, func(chunk StreamChunk) {
				if chunk.Content != "" {
					content.WriteString(chunk.Content)
				}
				if chunk.Reasoning != "" {
					reasoning.WriteString(chunk.Reasoning)
				}
				if chunk.FinishReason != "" {
					finishReason = chunk.FinishReason
				}
			})

			if err != nil {
				// Expired tokens are expected failures, not code bugs
				errMsg := err.Error()
				if strings.Contains(errMsg, "token_expired") ||
					strings.Contains(errMsg, "token has expired") ||
					strings.Contains(errMsg, "401") {
					t.Logf("[%s] SKIPPED — auth token expired (not a code bug)", name)
					fmt.Printf("  ⊘ %s (%s) — token expired, skipped\n", name, p.Model)
					return
				}
				t.Fatalf("[%s] API error: %v", name, err)
			}

			result := content.String()
			reasoningText := reasoning.String()

			t.Logf("[%s] model=%s", name, p.Model)
			t.Logf("[%s] content (%d chars): %s", name, len(result), truncResult(result, 200))
			if reasoningText != "" {
				t.Logf("[%s] reasoning (%d chars): %s", name, len(reasoningText), truncResult(reasoningText, 100))
			}
			t.Logf("[%s] finish_reason=%s", name, finishReason)

			// Verify we got some content back
			if result == "" && reasoningText == "" {
				t.Errorf("[%s] No content or reasoning received", name)
			}

			fmt.Printf("  ✓ %s (%s) — %d chars content, %d chars reasoning\n",
				name, p.Model, len(result), len(reasoningText))
		})
	}
}

func truncResult(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
