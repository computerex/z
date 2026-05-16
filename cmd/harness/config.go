package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Config holds the active runtime configuration.
type Config struct {
	Provider            string  `json:"provider"`
	APIBase             string  `json:"api_url"`
	APIKey              string  `json:"api_key"`
	Model               string  `json:"model"`
	MaxTokens           int     `json:"max_tokens"`
	MaxContextTokens    int     `json:"max_context_tokens"`
	Temperature         float64 `json:"temperature"`
	CompactionThreshold float64 `json:"compaction_threshold"`
	ReasoningEffort     string  `json:"reasoning_effort"`
	Providers           map[string]ProviderConfig          `json:"-"`
	MCP                 map[string]map[string]interface{}  `json:"-"`
}

// ProviderConfig holds a named provider profile.
type ProviderConfig struct {
	APIURL              string  `json:"api_url"`
	APIKey              string  `json:"api_key"`
	Model               string  `json:"model"`
	MaxTokens           int     `json:"max_tokens"`
	Temperature         float64 `json:"temperature"`
	CompactionThreshold float64 `json:"compaction_threshold"`
}

// zJSON is the raw ~/.z.json format.
type zJSON struct {
	APIURL              string                    `json:"api_url"`
	APIKey              string                    `json:"api_key"`
	Model               string                    `json:"model"`
	MaxTokens           int                       `json:"max_tokens"`
	Temperature         *float64                  `json:"temperature"`
	CompactionThreshold *float64                  `json:"compaction_threshold"`
	ReasoningEffort     string                    `json:"reasoning_effort"`
	Providers           map[string]ProviderConfig `json:"providers"`
	MCP                 map[string]json.RawMessage `json:"mcp"`
	ModelHistory        []ModelHistoryEntry       `json:"model_history"`
}

// ModelHistoryEntry tracks MRU model selections.
type ModelHistoryEntry struct {
	Model   string `json:"model"`
	Profile string `json:"profile"`
}

func defaultConfig() *Config {
	return &Config{
		Provider:            "openai",
		Model:               "gpt-4o",
		MaxTokens:           4096,
		MaxContextTokens:    128000,
		Temperature:         0.7,
		CompactionThreshold: 0.85,
		ReasoningEffort:     "high",
		Providers:           make(map[string]ProviderConfig),
		MCP:                 make(map[string]map[string]interface{}),
	}
}

func globalConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".z.json")
}

// loadConfig reads ~/.z.json and returns the active Config.
func loadConfig() *Config {
	cfg := defaultConfig()

	data, err := os.ReadFile(globalConfigPath())
	if err != nil {
		return cfg
	}

	var raw zJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return cfg
	}

	if raw.APIURL != "" {
		cfg.APIBase = raw.APIURL
	}
	if raw.APIKey != "" {
		cfg.APIKey = raw.APIKey
	}
	if raw.Model != "" {
		cfg.Model = raw.Model
	}
	if raw.MaxTokens > 0 {
		cfg.MaxTokens = raw.MaxTokens
	}
	if raw.Temperature != nil {
		cfg.Temperature = *raw.Temperature
	}
	if raw.CompactionThreshold != nil {
		cfg.CompactionThreshold = *raw.CompactionThreshold
	}
	if raw.ReasoningEffort != "" {
		cfg.ReasoningEffort = raw.ReasoningEffort
	}

	if raw.Providers != nil {
		cfg.Providers = raw.Providers
	}

	// Parse MCP config
	if raw.MCP != nil {
		for name, rawData := range raw.MCP {
			var m map[string]interface{}
			if json.Unmarshal(rawData, &m) == nil {
				cfg.MCP[name] = m
			}
		}
	}

	// Detect provider name from API URL
	if cfg.APIBase != "" {
		switch {
		case strings.Contains(cfg.APIBase, "copilot"):
			cfg.Provider = "copilot"
		case strings.Contains(cfg.APIBase, "anthropic"):
			cfg.Provider = "anthropic"
		case strings.Contains(cfg.APIBase, "bedrock"):
			cfg.Provider = "bedrock"
		case strings.Contains(cfg.APIBase, "openai"):
			cfg.Provider = "openai"
		default:
			cfg.Provider = "openai-compatible"
		}
	}

	// If no active config but providers exist, bootstrap from first provider
	if (cfg.APIBase == "" || cfg.APIKey == "") && len(cfg.Providers) > 0 {
		for name, p := range cfg.Providers {
			if p.APIURL != "" {
				cfg.APIBase = p.APIURL
			}
			if p.APIKey != "" {
				cfg.APIKey = p.APIKey
			}
			if p.Model != "" {
				cfg.Model = p.Model
			}
			if p.MaxTokens > 0 {
				cfg.MaxTokens = p.MaxTokens
			}
			cfg.Provider = name
			break
		}
	}

	return cfg
}

// saveConfig writes current top-level config fields back to ~/.z.json, preserving other fields.
func saveConfig(cfg *Config) error {
	return saveActiveConfigFields(map[string]interface{}{
		"api_url":     cfg.APIBase,
		"api_key":     cfg.APIKey,
		"model":       cfg.Model,
		"max_tokens":  cfg.MaxTokens,
		"temperature": cfg.Temperature,
	})
}

// saveActiveConfigFields writes specific top-level fields to ~/.z.json (read-modify-write).
func saveActiveConfigFields(updates map[string]interface{}) error {
	path := globalConfigPath()
	existing := make(map[string]interface{})

	data, err := os.ReadFile(path)
	if err == nil {
		json.Unmarshal(data, &existing)
	}

	for k, v := range updates {
		existing[k] = v
	}

	out, err := json.MarshalIndent(existing, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0644)
}

// saveProviderProfileField updates a provider profile's field in ~/.z.json.
func saveProviderProfileField(profile string, field string, value interface{}) error {
	path := globalConfigPath()
	existing := make(map[string]interface{})

	data, err := os.ReadFile(path)
	if err == nil {
		json.Unmarshal(data, &existing)
	}

	providers, _ := existing["providers"].(map[string]interface{})
	if providers == nil {
		providers = make(map[string]interface{})
	}

	prof, _ := providers[profile].(map[string]interface{})
	if prof == nil {
		prof = make(map[string]interface{})
	}

	prof[field] = value
	providers[profile] = prof
	existing["providers"] = providers

	out, err := json.MarshalIndent(existing, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0644)
}

const modelHistoryMax = 20

// recordModelHistory pushes a model+profile to the front of model_history in ~/.z.json.
func recordModelHistory(model, profile string) {
	path := globalConfigPath()
	existing := make(map[string]interface{})

	data, err := os.ReadFile(path)
	if err == nil {
		json.Unmarshal(data, &existing)
	}

	// Load existing history
	var history []ModelHistoryEntry
	if rawHist, ok := existing["model_history"]; ok {
		if arr, ok := rawHist.([]interface{}); ok {
			for _, item := range arr {
				if m, ok := item.(map[string]interface{}); ok {
					history = append(history, ModelHistoryEntry{
						Model:   fmt.Sprintf("%v", m["model"]),
						Profile: fmt.Sprintf("%v", m["profile"]),
					})
				}
			}
		}
	}

	// Remove existing entry for same model+profile
	filtered := make([]ModelHistoryEntry, 0, len(history))
	for _, h := range history {
		if !(h.Model == model && h.Profile == profile) {
			filtered = append(filtered, h)
		}
	}

	// Prepend new entry
	filtered = append([]ModelHistoryEntry{{Model: model, Profile: profile}}, filtered...)
	if len(filtered) > modelHistoryMax {
		filtered = filtered[:modelHistoryMax]
	}

	// Convert back to interface slice for JSON
	histSlice := make([]interface{}, len(filtered))
	for i, h := range filtered {
		histSlice[i] = map[string]interface{}{
			"model":   h.Model,
			"profile": h.Profile,
		}
	}
	existing["model_history"] = histSlice

	out, _ := json.MarshalIndent(existing, "", "  ")
	os.WriteFile(path, out, 0644)
}

// loadModelHistory reads the model_history from ~/.z.json.
func loadModelHistory() []ModelHistoryEntry {
	data, err := os.ReadFile(globalConfigPath())
	if err != nil {
		return nil
	}

	var raw map[string]interface{}
	if json.Unmarshal(data, &raw) != nil {
		return nil
	}

	arr, ok := raw["model_history"].([]interface{})
	if !ok {
		return nil
	}

	var history []ModelHistoryEntry
	for _, item := range arr {
		if m, ok := item.(map[string]interface{}); ok {
			history = append(history, ModelHistoryEntry{
				Model:   fmt.Sprintf("%v", m["model"]),
				Profile: fmt.Sprintf("%v", m["profile"]),
			})
		}
	}
	return history
}

// detectProviderLabel returns a human-readable label for an API URL.
func detectProviderLabel(apiURL string) string {
	u := strings.ToLower(apiURL)
	switch {
	case strings.Contains(u, "copilot") || strings.Contains(u, "githubcopilot"):
		return "GitHub Copilot"
	case strings.Contains(u, "openai.com"):
		return "OpenAI"
	case strings.Contains(u, "anthropic"):
		return "Anthropic"
	case strings.Contains(u, "bedrock") && strings.Contains(u, "amazonaws"):
		return "AWS Bedrock"
	case strings.Contains(u, "openrouter"):
		return "OpenRouter"
	case strings.Contains(u, "together"):
		return "Together AI"
	case strings.Contains(u, "z.ai"):
		return "Z.AI (GLM)"
	case strings.Contains(u, "nano-gpt") || strings.Contains(u, "nanogpt"):
		return "NanoGPT"
	case strings.Contains(u, "ollama") || strings.Contains(u, "localhost:11434"):
		return "Ollama"
	default:
		return "OpenAI-compatible"
	}
}

// getSessionsDir returns the session storage directory for a workspace.
func getSessionsDir(workspace string) string {
	home, _ := os.UserHomeDir()
	h := sha256.Sum256([]byte(workspace))
	hash := fmt.Sprintf("%x", h[:8])
	return filepath.Join(home, ".z", "sessions", hash)
}

func getSessionPath(workspace, name string) string {
	dir := getSessionsDir(workspace)
	os.MkdirAll(dir, 0755)
	return filepath.Join(dir, name+".json")
}

func listAllSessions(workspace string) []string {
	dir := getSessionsDir(workspace)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var sessions []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			name := strings.TrimSuffix(e.Name(), ".json")
			sessions = append(sessions, name)
		}
	}
	return sessions
}

func runInstallWizard() {
	fmt.Println("\n  Setup wizard")
	fmt.Println()

	cfg := defaultConfig()

	fmt.Print("  API URL: ")
	fmt.Scanln(&cfg.APIBase)
	fmt.Print("  API Key: ")
	fmt.Scanln(&cfg.APIKey)
	fmt.Print("  Model [gpt-4o]: ")
	var model string
	fmt.Scanln(&model)
	if model != "" {
		cfg.Model = model
	}

	if err := saveConfig(cfg); err != nil {
		fmt.Printf("  Error saving config: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  \033[32m✓\033[0m Config saved to %s\n\n", globalConfigPath())
}
