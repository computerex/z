package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
)

type App struct {
	workspace         string
	config            *Config
	costTracker       *CostTracker
	sessionName       string
	sessionManager    *SessionManager
	tools             *ToolRegistry
	messages          []Message
	context           *ContextContainer
	duplicateDetector *DuplicateDetector
}

func NewApp(workspace string, config *Config) (*App, error) {
	exePath, err := os.Executable()
	if err != nil {
		exePath = "."
	}
	harnessDir := filepath.Dir(exePath)

	sessionsDir := filepath.Join(harnessDir, ".sessions")
	sessionManager, err := NewSessionManager(sessionsDir)
	if err != nil {
		return nil, err
	}

	return &App{
		workspace:         workspace,
		config:           config,
		costTracker:       NewCostTracker(),
		sessionName:       "default",
		sessionManager:    sessionManager,
		tools:             NewToolRegistry(workspace),
		messages:          []Message{},
		context:           NewContextContainer(),
		duplicateDetector: NewDuplicateDetector(),
	}, nil
}

func (a *App) Run() error {
	a.messages = []Message{
		{Role: "system", Content: getSystemPrompt(a.workspace)},
	}

	if err := a.loadSession("default"); err == nil {
		msgCount := len(a.messages) - 1
		fmt.Printf("Resumed session 'default' (%d messages)\n", msgCount)
	} else {
		fmt.Println("New session 'default'")
	}

	fmt.Println("\nHarness - /help for commands")
	fmt.Printf("Workspace: %s\n\n", a.workspace)

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print(a.getPrompt())
		input, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		input = strings.TrimSpace(input)

		if input == "" {
			continue
		}

		if strings.HasPrefix(input, "/") {
			if err := a.handleCommand(input); err != nil {
				fmt.Printf("Error: %v\n", err)
			}
			continue
		}

		if err := a.processRequest(input); err != nil {
			fmt.Printf("Error: %v\n", err)
		}

		a.saveSession(a.sessionName)
		fmt.Println()
	}

	return nil
}

func (a *App) RunSingle(userInput string) error {
	if len(a.messages) == 0 {
		a.messages = []Message{
			{Role: "system", Content: getSystemPrompt(a.workspace)},
		}
	}

	a.messages = append(a.messages, Message{Role: "user", Content: userInput})

	if err := a.processRequest(userInput); err != nil {
		return err
	}

	a.saveSession(a.sessionName)
	return nil
}

func (a *App) getPrompt() string {
	tokenInfo := ""
	tokens := EstimateMessagesTokens(a.messages)
	if tokens > 1000 {
		tokenInfo = fmt.Sprintf(" [%dk]", tokens/1000)
	}
	return fmt.Sprintf("[%s]%s> ", a.sessionName, tokenInfo)
}

func (a *App) handleCommand(cmd string) error {
	parts := strings.SplitN(cmd, " ", 2)
	command := strings.ToLower(parts[0])
	arg := ""
	if len(parts) > 1 {
		arg = strings.TrimSpace(parts[1])
	}

	switch command {
	case "/exit", "/quit", "/q":
		a.saveSession(a.sessionName)
		fmt.Println("Session saved. Exiting...")
		os.Exit(0)

	case "/sessions":
		return a.listSessions()

	case "/session":
		if arg == "" {
			fmt.Printf("Current: %s\n", a.sessionName)
			return nil
		}
		return a.switchSession(arg)

	case "/clear":
		a.messages = []Message{{Role: "system", Content: getSystemPrompt(a.workspace)}}
		a.context.Clear()
		a.duplicateDetector.Clear()
		a.costTracker.Reset()
		fmt.Println("History cleared.")

	case "/save":
		a.saveSession(a.sessionName)
		fmt.Printf("Saved '%s'\n", a.sessionName)

	case "/delete":
		return a.deleteSession(arg)

	case "/history":
		fmt.Printf("Messages: %d\n", len(a.messages))

	case "/tokens":
		return a.showTokenBreakdown()

	case "/compact":
		return a.compactHistory(arg)

	case "/ctx":
		stats := a.getContextStats()
		fmt.Printf("Context: %d tokens (%d%% of %d limit)\n", stats["tokens"], int(stats["percent"].(float64)), stats["max_allowed"])
		fmt.Printf("Messages: %d | Context items: %d\n", stats["messages"], stats["context_items"])
		fmt.Printf("%s\n", a.context.Summary())

	case "/config":
		keyPreview := a.config.APIKey
		if len(keyPreview) > 8 {
			keyPreview = keyPreview[:8] + "..."
		}
		fmt.Printf("API URL: %s\n", a.config.APIURL)
		fmt.Printf("Model:   %s\n", a.config.Model)
		fmt.Printf("API Key: %s\n", keyPreview)
		fmt.Printf("Max tokens: %d\n", a.config.MaxTokens)

	case "/help", "/?":
		return a.showHelp()

	default:
		fmt.Println("Type /help for commands")
	}

	return nil
}

func (a *App) listSessions() error {
	sessions, err := a.sessionManager.List(a.workspace)
	if err != nil {
		return err
	}

	if len(sessions) == 0 {
		fmt.Println("No sessions.")
		return nil
	}

	fmt.Println("Sessions:")
	for _, s := range sessions {
		marker := " "
		if s.Name == a.sessionName {
			marker = "*"
		}
		fmt.Printf(" %s %-20s (%d msgs)\n", marker, s.Name, s.MessageCount)
	}

	return nil
}

func (a *App) switchSession(name string) error {
	a.saveSession(a.sessionName)
	a.sessionName = name
	if err := a.loadSession(name); err == nil {
		msgCount := len(a.messages) - 1
		fmt.Printf("Switched to '%s' (%d msgs)\n", a.sessionName, msgCount)
	} else {
		a.messages = []Message{{Role: "system", Content: getSystemPrompt(a.workspace)}}
		a.context.Clear()
		fmt.Printf("Created new session '%s'\n", a.sessionName)
	}

	return nil
}

func (a *App) deleteSession(name string) error {
	if name == "" {
		fmt.Println("Usage: /delete <session_name>")
		return nil
	}

	if name == a.sessionName {
		fmt.Println("Cannot delete active session.")
		return nil
	}

	if err := a.sessionManager.Delete(a.workspace, name); err != nil {
		return fmt.Errorf("session '%s' not found", name)
	}

	fmt.Printf("Deleted '%s'\n", name)
	return nil
}

func (a *App) showTokenBreakdown() error {
	systemTokens := 0
	convTokens := 0

	for _, msg := range a.messages {
		tokens := EstimateTokens(msg.Content)
		if msg.Role == "system" {
			systemTokens += tokens
		} else {
			convTokens += tokens
		}
	}

	fmt.Printf("Token breakdown:\n")
	fmt.Printf("  System prompt: %d tokens\n", systemTokens)
	fmt.Printf("  Conversation:  %d tokens (%d messages)\n", convTokens, len(a.messages)-1)
	fmt.Printf("  Total:         %d tokens\n", systemTokens+convTokens)

	return nil
}

func (a *App) compactHistory(strategy string) error {
	if strategy == "" {
		strategy = "half"
	}

	before := EstimateMessagesTokens(a.messages)
	result := TruncateConversation(a.messages, strategy)
	a.messages = result.Messages
	after := EstimateMessagesTokens(a.messages)

	fmt.Printf("Compacted: %d -> %d tokens (-%d)\n", before, after, before-after)
	return nil
}

func (a *App) showHelp() error {
	fmt.Println(`Commands:
  /sessions          - List all sessions
  /session <name>    - Switch to session (creates if new)
  /delete <name>     - Delete a session
  /clear             - Clear conversation history
  /compact [strat]   - Remove older messages (half/quarter/last2)
  /tokens            - Show token breakdown
  /config            - Show current API configuration
  /save              - Save current session
  /history           - Show message count
  /ctx               - Show context container
  /exit              - Save and exit
[/dim]`)
	return nil
}

func (a *App) processRequest(userInput string) error {
	a.messages = append(a.messages, Message{Role: "user", Content: userInput})

	_, maxAllowed := GetModelLimits(a.config.Model)
	tokenCount := EstimateMessagesTokens(a.messages)
	if tokenCount > maxAllowed {
		strategy := "half"
		if tokenCount > maxAllowed*3/2 {
			strategy = "quarter"
		}
		result := TruncateConversation(a.messages, strategy)
		a.messages = result.Messages
		fmt.Printf("[!] Context truncated: removed %d messages (%s)\n", result.RemovedCount, strategy)
	}

	client := NewAPIClient(a.config.APIKey, a.config.APIURL, a.config.Model)
	client.Temperature = a.config.Temperature
	client.MaxTokens = a.config.MaxTokens

	fmt.Println()
	fullContent := ""
	toolCalls := 0

	onChunk := func(chunk string) {
		cleanChunk := Render(chunk)
		fullContent += cleanChunk
		fmt.Print(cleanChunk)
	}

	onInterrupt := func() bool {
		return false
	}

	response, err := client.ChatStream(a.messages, onChunk, onInterrupt)
	if err != nil {
		return err
	}

	fmt.Println()

	inputTokens := EstimateMessagesTokens(a.messages)
	outputTokens := EstimateTokens(fullContent)
	a.costTracker.RecordCall(a.config.Model, inputTokens, outputTokens, 0, toolCalls, response.FinishReason)

	maxIterations := 30
	for i := 0; i < maxIterations; i++ {
		tool := ParseXMLTool(fullContent)
		if tool == nil {
			if strings.Contains(fullContent, "<attempt_completion>") {
				match := regexp.MustCompile(`<result>(.*?)</result>`).FindStringSubmatch(fullContent)
				if match != nil {
					a.messages = append(a.messages, Message{Role: "assistant", Content: fullContent})
					return nil
				}
			}

			a.messages = append(a.messages, Message{Role: "assistant", Content: fullContent})
			return nil
		}

		toolCalls++
		fmt.Printf("> Executing: %s\n", tool.Name)
		toolResult := a.tools.ExecuteTool(tool)

		if !toolResult.Success {
			fmt.Printf("[X] %s\n", toolResult.Message)
		}

		// Build result message - include Content if present (e.g., file contents)
		resultContent := toolResult.Message
		if toolResult.Content != "" {
			resultContent = toolResult.Message + "\n" + toolResult.Content
		}

		a.messages = append(a.messages, Message{Role: "assistant", Content: fullContent})
		a.messages = append(a.messages, Message{Role: "user", Content: fmt.Sprintf("[%s result]\n%s", tool.Name, resultContent)})

		fullContent = ""
		onChunk = func(chunk string) {
			cleanChunk := Render(chunk)
			fullContent += cleanChunk
			fmt.Print(cleanChunk)
		}

		response, err = client.ChatStream(a.messages, onChunk, onInterrupt)
		if err != nil {
			return err
		}

		fmt.Println()

		inputTokens = EstimateMessagesTokens(a.messages)
		outputTokens = EstimateTokens(fullContent)
		a.costTracker.RecordCall(a.config.Model, inputTokens, outputTokens, 0, toolCalls, response.FinishReason)
	}

	return fmt.Errorf("max iterations reached")
}

func (a *App) saveSession(name string) error {
	return a.sessionManager.Save(a.workspace, name, a.messages, a.context.GetContextItems())
}

func (a *App) loadSession(name string) error {
	session, err := a.sessionManager.Load(a.workspace, name)
	if err != nil {
		return err
	}

	a.messages = session.Messages
	a.context.LoadContextItems(session.ContextItems)
	return nil
}

func (a *App) getContextStats() map[string]interface{} {
	_, maxAllowed := GetModelLimits(a.config.Model)
	tokens := EstimateMessagesTokens(a.messages)
	percent := 0.0
	if maxAllowed > 0 {
		percent = float64(tokens) / float64(maxAllowed) * 100
	}
	return map[string]interface{}{
		"tokens":        tokens,
		"max_allowed":   maxAllowed,
		"percent":       percent,
		"messages":      len(a.messages),
		"context_items": a.context.Count(),
		"context_chars": a.context.TotalSize(),
	}
}

func killProcess(proc *exec.Cmd) {
	if proc.Process != nil {
		if runtime.GOOS == "windows" {
			proc.SysProcAttr = &syscall.SysProcAttr{
				CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
			}
		}
		proc.Process.Kill()
	}
}

func CleanupBackgroundProcs() {
	bgProcsMu.Lock()
	defer bgProcsMu.Unlock()

	for _, proc := range bgProcs {
		if proc.Proc.ProcessState == nil {
			if runtime.GOOS == "windows" {
				cmd := exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprint(proc.Proc.Process.Pid))
				cmd.Run()
			} else {
				killProcess(proc.Proc)
			}
		}
	}
	bgProcs = make(map[int]*BackgroundProcess)
}
