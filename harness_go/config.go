package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Config holds API configuration
type Config struct {
	APIURL      string `json:"api_url"`
	APIKey      string `json:"api_key"`
	Model       string `json:"model"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int    `json:"max_tokens,omitempty"`
}

// DefaultConfig returns default configuration values
func DefaultConfig() *Config {
	return &Config{
		APIURL:      "",
		APIKey:      "",
		Model:       "glm-4.7",
		Temperature: 0.7,
		MaxTokens:   8192,
	}
}

// Validate checks if config has required fields
func (c *Config) Validate() error {
	if c.APIURL == "" {
		return fmt.Errorf("API URL is required")
	}
	if c.APIKey == "" {
		return fmt.Errorf("API key is required")
	}
	if c.Model == "" {
		c.Model = "glm-4.7"
	}
	// Ensure URL ends with /
	if !strings.HasSuffix(c.APIURL, "/") {
		c.APIURL = c.APIURL + "/"
	}
	return nil
}

// LoadConfig loads configuration from a file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &config, nil
}

// SaveConfig saves configuration to a file
func SaveConfig(path string, config *Config) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// FindConfig searches for config file in multiple locations
func FindConfig(workspace string) (*Config, error) {
	// Check workspace config
	workspaceConfig := filepath.Join(workspace, ".z", ".z.json")
	if _, err := os.Stat(workspaceConfig); err == nil {
		return LoadConfig(workspaceConfig)
	}

	// Check home directory config
	homeDir, err := os.UserHomeDir()
	if err == nil {
		homeConfig := filepath.Join(homeDir, ".z.json")
		if _, err := os.Stat(homeConfig); err == nil {
			return LoadConfig(homeConfig)
		}
	}

	// Check harness directory
	harnessConfig := filepath.Join(".", ".z.json")
	if _, err := os.Stat(harnessConfig); err == nil {
		return LoadConfig(harnessConfig)
	}

	return nil, fmt.Errorf("configuration file not found")
}