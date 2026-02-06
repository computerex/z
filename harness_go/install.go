package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Provider represents an LLM provider
type Provider struct {
	Name         string
	BaseURL      string
	DefaultModel string
	Options      []string
}

// Available providers
var providers = []Provider{
	{
		Name:         "Z.AI Coding Plan",
		BaseURL:      "https://api.z.ai/api/coding/paas/v4/",
		DefaultModel: "glm-4.7",
		Options:      []string{"1", "z.ai", "coding"},
	},
	{
		Name:         "Z.AI Standard API",
		BaseURL:      "https://api.z.ai/api/paas/v4/",
		DefaultModel: "glm-4.7",
		Options:      []string{"2", "z.ai", "standard"},
	},
	{
		Name:         "MiniMax",
		BaseURL:      "https://api.minimax.io/v1/",
		DefaultModel: "MiniMax-M2.1",
		Options:      []string{"3", "minimax"},
	},
	{
		Name:         "Custom OpenAI-compatible API",
		BaseURL:      "",
		DefaultModel: "gpt-4",
		Options:      []string{"4", "custom"},
	},
}

// handleInstall runs the installation/setup wizard
func handleInstall(args []string) {
	var apiURL, apiKey, model string
	globalConfig := true

	// Parse headless args
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--api-url":
			if i+1 < len(args) {
				apiURL = args[i+1]
				i++
			}
		case "--api-key":
			if i+1 < len(args) {
				apiKey = args[i+1]
				i++
			}
		case "--model":
			if i+1 < len(args) {
				model = args[i+1]
				i++
			}
		case "--workspace-config", "-w":
			globalConfig = false
		}
	}

	// Headless mode
	if apiURL != "" && apiKey != "" {
		runInstallHeadless(apiURL, apiKey, model, globalConfig)
		return
	}

	// Interactive mode
	runInstallInteractive()
}

// runInstallHeadless configures the harness in headless mode
func runInstallHeadless(apiURL, apiKey, model string, globalConfig bool) {
	if model == "" {
		model = "glm-4.7"
	}

	// Ensure URL ends with /
	if !strings.HasSuffix(apiURL, "/") {
		apiURL = apiURL + "/"
	}

	config := map[string]interface{}{
		"api_url": apiURL,
		"api_key": apiKey,
		"model":   model,
	}

	var configPath string
	if globalConfig {
		homeDir, _ := os.UserHomeDir()
		configPath = filepath.Join(homeDir, ".z.json")
	} else {
		wd, _ := os.Getwd()
		configDir := filepath.Join(wd, ".z")
		os.MkdirAll(configDir, 0755)
		configPath = filepath.Join(configDir, ".z.json")
	}

	data, _ := json.MarshalIndent(config, "", "  ")
	os.WriteFile(configPath, data, 0600)

	fmt.Println("Configuration saved to:", configPath)
	fmt.Printf("  URL:   %s\n", apiURL)
	fmt.Printf("  Model: %s\n", model)
	fmt.Printf("  Key:   %s...\n", apiKey[:min(10, len(apiKey))])
}

// runInstallInteractive runs the interactive setup wizard
func runInstallInteractive() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("  LLM Harness Setup")
	fmt.Println(strings.Repeat("=", 60) + "\n")

	fmt.Println("Select your LLM provider:\n")
	fmt.Println("  [1] Z.AI Coding Plan (recommended)")
	fmt.Println("      - https://api.z.ai/api/coding/paas/v4/\n")
	fmt.Println("  [2] Z.AI Standard API")
	fmt.Println("      - https://api.z.ai/api/paas/v4/\n")
	fmt.Println("  [3] MiniMax")
	fmt.Println("      - https://api.minimax.io/v1/\n")
	fmt.Println("  [4] Custom OpenAI-compatible API")
	fmt.Println("      - Enter your own URL\n")

	var provider Provider
	for {
		fmt.Print("Enter choice [1/2/3/4]: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			provider = providers[0]
		case "2":
			provider = providers[1]
		case "3":
			provider = providers[2]
		case "4":
			provider = providers[3]
		default:
			fmt.Println("Please enter 1, 2, 3, or 4.")
			continue
		}
		break
	}

	fmt.Printf("\nUsing %s: %s\n", provider.Name, provider.BaseURL)

	// Get API key
	var apiKey string
	fmt.Println("\nEnter your API key:\n")
	for {
		fmt.Print("API Key: ")
		input, _ := reader.ReadString('\n')
		apiKey = strings.TrimSpace(input)
		if apiKey != "" {
			break
		}
		fmt.Println("API key is required.")
	}

	// Get model
	fmt.Print("\nModel name (default: ", provider.DefaultModel, "): ")
	input, _ := reader.ReadString('\n')
	model := strings.TrimSpace(input)
	if model == "" {
		model = provider.DefaultModel
	}

	// Custom URL handling
	baseURL := provider.BaseURL
	if provider.Name == "Custom OpenAI-compatible API" {
		fmt.Print("\nEnter API base URL: ")
		for {
			input, _ := reader.ReadString('\n')
			baseURL = strings.TrimSpace(input)
			if baseURL != "" {
				break
			}
			fmt.Println("URL is required.")
		}
		if !strings.HasSuffix(baseURL, "/") {
			baseURL = baseURL + "/"
		}
	}

	// Choose save location
	fmt.Println("\nWhere to save configuration?\n")
	fmt.Println("  [1] Global (~/.z.json) - applies to all projects")
	fmt.Println("  [2] Workspace (.z/.z.json) - this project only\n")

	var globalConfig bool
	for {
		fmt.Print("Enter choice [1/2] (default: 1): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if input == "" || input == "1" {
			globalConfig = true
			break
		}
		if input == "2" {
			globalConfig = false
			break
		}
		fmt.Println("Please enter 1 or 2.")
	}

	// Save config
	config := map[string]interface{}{
		"api_url": baseURL,
		"api_key": apiKey,
		"model":   model,
	}

	var configPath string
	location := "global"
	if globalConfig {
		homeDir, _ := os.UserHomeDir()
		configPath = filepath.Join(homeDir, ".z.json")
	} else {
		wd, _ := os.Getwd()
		configDir := filepath.Join(wd, ".z")
		os.MkdirAll(configDir, 0755)
		configPath = filepath.Join(configDir, ".z.json")
		location = "workspace"
	}

	data, _ := json.MarshalIndent(config, "", "  ")
	os.WriteFile(configPath, data, 0600)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("  Setup Complete!")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("\nConfiguration saved to: %s\n", configPath)
	fmt.Printf("  Location: %s\n", location)
	fmt.Printf("  Provider: %s\n", provider.Name)
	fmt.Printf("  Model:    %s\n", model)
	fmt.Printf("  Key:      %s...\n", apiKey[:min(10, len(apiKey))])
	fmt.Println("\nRun './harness' to start.\n")
}