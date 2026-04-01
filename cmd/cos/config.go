package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// Paths rooted at ~/COS/
var (
	cosDir    string
	agentsDir string
	notifyDir string
	logsDir   string
	configFile string
	sessionFile string
)

func initPaths() {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	cosDir = filepath.Join(home, "COS")
	agentsDir = filepath.Join(cosDir, "agents")
	notifyDir = filepath.Join(cosDir, "notify")
	logsDir = filepath.Join(cosDir, "logs")
	configFile = filepath.Join(cosDir, "config.json")
	sessionFile = filepath.Join(cosDir, "session.json")
}

func ensureDirs() {
	for _, d := range []string{cosDir, agentsDir, notifyDir, logsDir} {
		os.MkdirAll(d, 0755)
	}
}

// COSConfig mirrors ~/COS/config.json
type COSConfig struct {
	Repos         map[string]string `json:"repos"`
	Engineer      string            `json:"engineer"`
	Model         string            `json:"model"`
	SubAgentModel string            `json:"sub_agent_model"`
	MaxTurns      int               `json:"max_turns"`
	SubMaxTurns   int               `json:"sub_agent_max_turns"`
	TickInterval  int               `json:"tick_interval"` // seconds, 0 = default
	JiraProject   string            `json:"jira_project"`
	JiraBaseURL   string            `json:"jira_base_url"`
	GithubOrg     string            `json:"github_org"`
	GithubRepos   map[string]string `json:"github_repos"`
}

func loadConfig() COSConfig {
	cfg := COSConfig{
		Model:        "",
		MaxTurns:     200,
		SubMaxTurns:  100,
		TickInterval: 3600,
	}
	data, err := os.ReadFile(configFile)
	if err != nil {
		return cfg
	}
	json.Unmarshal(data, &cfg)
	if cfg.TickInterval <= 0 {
		cfg.TickInterval = 3600
	}
	return cfg
}

// Session tracks the COS brain conversation
type Session struct {
	SessionID string `json:"session_id"`
}

func loadSession() Session {
	var s Session
	data, err := os.ReadFile(sessionFile)
	if err != nil {
		return s
	}
	json.Unmarshal(data, &s)
	return s
}

func saveSession(s Session) {
	data, _ := json.MarshalIndent(s, "", "  ")
	os.WriteFile(sessionFile, data, 0644)
}

func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339)
}

func isWindows() bool {
	return runtime.GOOS == "windows"
}
