package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// ModelPricing defines pricing per model
type ModelPricing struct {
	Input  float64 // per 1M tokens
	Output float64 // per 1M tokens
}

// DefaultPricing holds default pricing for known models
var DefaultPricing = map[string]ModelPricing{
	// Z.AI GLM models
	"glm-4.7":   {Input: 0.50, Output: 1.50},
	"glm-4":     {Input: 0.50, Output: 1.50},
	"glm-4-plus": {Input: 1.00, Output: 3.00},
	"glm-4.6v":  {Input: 0.50, Output: 1.50}, // Vision model
	// MiniMax models
	"minimax-m2.1": {Input: 0.14, Output: 0.56},
	"minimax-text-01": {Input: 0.10, Output: 0.40},
	"abab6.5s-chat": {Input: 0.01, Output: 0.01},
	// Default fallback
	"default": {Input: 0.50, Output: 1.50},
}

// APICall represents a single API call record
type APICall struct {
	Timestamp     time.Time `json:"timestamp"`
	Model         string    `json:"model"`
	InputTokens   int       `json:"input_tokens"`
	OutputTokens  int       `json:"output_tokens"`
	TotalTokens   int       `json:"total_tokens"`
	InputCost     float64   `json:"input_cost"`
	OutputCost    float64   `json:"output_cost"`
	TotalCost     float64   `json:"total_cost"`
	DurationMs    float64   `json:"duration_ms"`
	ToolCalls     int       `json:"tool_calls"`
	FinishReason  string    `json:"finish_reason"`
}

// ToDict converts to a map
func (ac *APICall) ToDict() map[string]interface{} {
	return map[string]interface{}{
		"timestamp":     ac.Timestamp.Format(time.RFC3339),
		"model":         ac.Model,
		"input_tokens":   ac.InputTokens,
		"output_tokens":  ac.OutputTokens,
		"total_tokens":   ac.TotalTokens,
		"input_cost":     ac.InputCost,
		"output_cost":    ac.OutputCost,
		"total_cost":     ac.TotalCost,
		"duration_ms":   ac.DurationMs,
		"tool_calls":    ac.ToolCalls,
		"finish_reason": ac.FinishReason,
	}
}

// CostSummary holds a summary of costs
type CostSummary struct {
	TotalCalls        int     `json:"total_calls"`
	TotalInputTokens  int     `json:"total_input_tokens"`
	TotalOutputTokens int     `json:"total_output_tokens"`
	TotalTokens       int     `json:"total_tokens"`
	TotalInputCost    float64 `json:"total_input_cost"`
	TotalOutputCost   float64 `json:"total_output_cost"`
	TotalCost         float64 `json:"total_cost"`
	TotalDurationMs   float64 `json:"total_duration_ms"`
	TotalToolCalls    int     `json:"total_tool_calls"`
}

// ToDict converts to a map
func (cs *CostSummary) ToDict() map[string]interface{} {
	avgTokens := 0.0
	avgCost := 0.0
	if cs.TotalCalls > 0 {
		avgTokens = float64(cs.TotalTokens) / float64(cs.TotalCalls)
		avgCost = cs.TotalCost / float64(cs.TotalCalls)
	}
	return map[string]interface{}{
		"total_calls":         cs.TotalCalls,
		"total_input_tokens":  cs.TotalInputTokens,
		"total_output_tokens": cs.TotalOutputTokens,
		"total_tokens":       cs.TotalTokens,
		"total_input_cost":   round(cs.TotalInputCost, 6),
		"total_output_cost":  round(cs.TotalOutputCost, 6),
		"total_cost":         round(cs.TotalCost, 6),
		"total_duration_ms":  round(cs.TotalDurationMs, 2),
		"total_tool_calls":   cs.TotalToolCalls,
		"avg_tokens_per_call": round(avgTokens, 1),
		"avg_cost_per_call":   round(avgCost, 6),
	}
}

// FormatHuman returns a human-readable summary
func (cs *CostSummary) FormatHuman() string {
	return fmt.Sprintf(
		"API Calls: %d\nTokens: %d in / %d out (%d total)\nCost: $%.4f in / $%.4f out ($%.4f total)\nTool Calls: %d\nDuration: %.2fs",
		cs.TotalCalls,
		cs.TotalInputTokens,
		cs.TotalOutputTokens,
		cs.TotalTokens,
		cs.TotalInputCost,
		cs.TotalOutputCost,
		cs.TotalCost,
		cs.TotalToolCalls,
		cs.TotalDurationMs/1000,
	)
}

// CostTracker tracks API costs
type CostTracker struct {
	pricing             map[string]ModelPricing
	calls               []*APICall
	sessionStart        time.Time
	lastReportCallCount int
}

// NewCostTracker creates a new cost tracker
func NewCostTracker() *CostTracker {
	pricing := make(map[string]ModelPricing)
	for k, v := range DefaultPricing {
		pricing[k] = v
	}
	return &CostTracker{
		pricing:      pricing,
		calls:        make([]*APICall, 0),
		sessionStart: time.Now(),
	}
}

// GetPricing gets pricing for a model
func (ct *CostTracker) GetPricing(model string) ModelPricing {
	// Exact match
	if pricing, ok := ct.pricing[model]; ok {
		return pricing
	}

	// Case-insensitive match
	modelLower := strings.ToLower(model)
	for key, pricing := range ct.pricing {
		if strings.ToLower(key) == modelLower {
			return pricing
		}
	}

	// Partial match
	for key, pricing := range ct.pricing {
		if strings.Contains(modelLower, strings.ToLower(key)) {
			return pricing
		}
	}

	return ct.pricing["default"]
}

// RecordCall records an API call
func (ct *CostTracker) RecordCall(model string, inputTokens, outputTokens int, durationMs float64, toolCalls int, finishReason string) *APICall {
	pricing := ct.GetPricing(model)

	// Cost per million tokens
	inputCost := (float64(inputTokens) / 1_000_000) * pricing.Input
	outputCost := (float64(outputTokens) / 1_000_000) * pricing.Output

	call := &APICall{
		Timestamp:     time.Now(),
		Model:         model,
		InputTokens:   inputTokens,
		OutputTokens:  outputTokens,
		TotalTokens:   inputTokens + outputTokens,
		InputCost:     inputCost,
		OutputCost:    outputCost,
		TotalCost:     inputCost + outputCost,
		DurationMs:    durationMs,
		ToolCalls:     toolCalls,
		FinishReason:  finishReason,
	}

	ct.calls = append(ct.calls, call)
	return call
}

// GetSummary returns a summary
func (ct *CostTracker) GetSummary() *CostSummary {
	summary := &CostSummary{}

	for _, call := range ct.calls {
		summary.TotalCalls++
		summary.TotalInputTokens += call.InputTokens
		summary.TotalOutputTokens += call.OutputTokens
		summary.TotalTokens += call.TotalTokens
		summary.TotalInputCost += call.InputCost
		summary.TotalOutputCost += call.OutputCost
		summary.TotalCost += call.TotalCost
		summary.TotalDurationMs += call.DurationMs
		summary.TotalToolCalls += call.ToolCalls
	}

	return summary
}

// GetLastCall returns the most recent call
func (ct *CostTracker) GetLastCall() *APICall {
	if len(ct.calls) == 0 {
		return nil
	}
	return ct.calls[len(ct.calls)-1]
}

// Reset resets the tracker
func (ct *CostTracker) Reset() {
	ct.calls = make([]*APICall, 0)
	ct.sessionStart = time.Now()
	ct.lastReportCallCount = 0
}

// ToDict exports all data
func (ct *CostTracker) ToDict() map[string]interface{} {
	return map[string]interface{}{
		"session_start": ct.sessionStart.Format(time.RFC3339),
		"summary":       ct.GetSummary().ToDict(),
		"calls": func() []map[string]interface{} {
			calls := make([]map[string]interface{}, len(ct.calls))
			for i, call := range ct.calls {
				calls[i] = call.ToDict()
			}
			return calls
		}(),
	}
}

// Save saves tracking data to file
func (ct *CostTracker) Save(path string) error {
	data, err := json.MarshalIndent(ct.ToDict(), "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// GetTotalCalls returns total call count
func (ct *CostTracker) GetTotalCalls() int {
	return len(ct.calls)
}

// GetTotalCost returns total cost
func (ct *CostTracker) GetTotalCost() float64 {
	return ct.GetSummary().TotalCost
}

func round(v float64, precision int) float64 {
	factor := float64(1)
	for i := 0; i < precision; i++ {
		factor *= 10
	}
	return math.Round(v*factor) / factor
}