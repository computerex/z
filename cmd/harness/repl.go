package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/chzyer/readline"
)

// REPL handles the interactive read-eval-print loop.
type REPL struct {
	Workspace   string
	Config      *Config
	Agent       *Agent
	SessionPath string
	rl          *readline.Instance
}

// NewREPL creates a new interactive REPL.
func NewREPL(workspace string, cfg *Config, agent *Agent, sessionPath string) *REPL {
	return &REPL{
		Workspace:   workspace,
		Config:      cfg,
		Agent:       agent,
		SessionPath: sessionPath,
	}
}

// Run starts the interactive loop.
func (r *REPL) Run() error {
	var err error
	r.rl, err = readline.NewEx(&readline.Config{
		Prompt:          "\033[1;36m❯\033[0m ",
		HistoryFile:     filepath.Join(os.TempDir(), ".harness_history"),
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	})
	if err != nil {
		return fmt.Errorf("readline init: %w", err)
	}
	defer r.rl.Close()

	for {
		r.rl.SetPrompt(r.buildPrompt())
		line, err := r.rl.Readline()
		if err == readline.ErrInterrupt {
			continue
		}
		if err == io.EOF {
			fmt.Println("Goodbye!")
			return nil
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Slash commands
		if strings.HasPrefix(line, "/") {
			handled, exit := r.handleSlashCommand(line)
			if exit {
				return nil
			}
			if handled {
				continue
			}
		}

		// Shell command shortcut
		if strings.HasPrefix(line, "!") {
			cmd := strings.TrimPrefix(line, "!")
			cmd = strings.TrimSpace(cmd)
			if cmd != "" {
				result := r.Agent.Tools.executeCommand(map[string]string{"command": cmd})
				if result.Output != "" {
					fmt.Println(result.Output)
				}
				if result.Error != "" {
					fmt.Fprintf(os.Stderr, "Error: %s\n", result.Error)
				}
			}
			continue
		}

		// Regular message — send to agent
		resetInterrupt()
		result := r.Agent.RunMessage(line)
		_ = result // Already displayed by the agent
	}
}

// buildPrompt returns the input prompt with context/iter info.
func (r *REPL) buildPrompt() string {
	if r.Agent.Session == nil {
		return "\033[1;36m❯\033[0m "
	}
	tokens := r.Agent.Session.EstimateTokens()
	maxTokens := r.Agent.Config.MaxContextTokens
	if maxTokens == 0 {
		maxTokens = 128000
	}
	msgs := len(r.Agent.Session.Messages)
	if msgs > 0 {
		msgs-- // don't count system prompt
	}
	pct := 0
	if maxTokens > 0 {
		pct = tokens * 100 / maxTokens
	}
	return fmt.Sprintf("\033[2m[%dk/%dk %d%%  %d msgs]\033[0m \033[1;36m❯\033[0m ", tokens/1000, maxTokens/1000, pct, msgs)
}

// handleSlashCommand processes a / prefixed command.
// Returns (handled, shouldExit).
func (r *REPL) handleSlashCommand(line string) (bool, bool) {
	parts := strings.SplitN(line, " ", 2)
	cmd := strings.ToLower(parts[0])
	args := ""
	if len(parts) > 1 {
		args = strings.TrimSpace(parts[1])
	}

	switch cmd {
	case "/exit", "/quit", "/q":
		fmt.Println("Goodbye!")
		return true, true

	case "/help", "/h":
		r.showHelp()

	case "/new":
		r.newSession(args)

	case "/sessions", "/ls":
		r.listSessions()

	case "/session":
		r.switchSession(args)

	case "/delete":
		r.deleteSession(args)

	case "/clear":
		r.clearSession()

	case "/save":
		r.saveSession(args)

	case "/history":
		r.showHistory()

	case "/cost":
		fmt.Println("Cost tracking not yet implemented in Go port")

	case "/ctx", "/context":
		r.showContext()

	case "/tokens":
		tokens := r.Agent.Session.EstimateTokens()
		fmt.Printf("Estimated tokens: %d\n", tokens)

	case "/compact":
		r.Agent.compactContext()
		fmt.Println("Context compacted.")

	case "/todo", "/todos":
		fmt.Println("Todo management not yet implemented in Go port")

	case "/config":
		r.showConfig()

	case "/providers":
		if args != "" {
			r.handleProvidersSubcommand(args)
		} else {
			r.showProviders()
		}

	case "/model":
		if args != "" {
			r.handleModelSwitch(args)
		} else {
			r.showModelInfo()
		}

	case "/maxctx":
		if args != "" {
			var n int
			fmt.Sscanf(args, "%d", &n)
			if n > 0 {
				r.Agent.Config.MaxContextTokens = n
				fmt.Printf("Max context tokens set to: %d\n", n)
			}
		} else {
			fmt.Printf("Max context tokens: %d\n", r.Agent.Config.MaxContextTokens)
		}

	case "/undo":
		fmt.Println("Undo not yet implemented in Go port")

	case "/redo":
		fmt.Println("Redo not yet implemented in Go port")

	case "/bg":
		r.showBackgroundProcesses()

	case "/log":
		r.showLog(args)

	case "/iter":
		r.handleIter(args)

	case "/mcp":
		r.showMCP()

	default:
		fmt.Printf("Unknown command: %s (type /help for available commands)\n", cmd)
	}

	return true, false
}

func (r *REPL) showHelp() {
	fmt.Print(`
Commands:
  /help, /h          Show this help
  /exit, /quit, /q   Exit harness
  /new [name]        Start a new session
  /sessions, /ls     List saved sessions
  /session <name>    Load a session
  /delete <name>     Delete a session
  /clear             Clear current session
  /save [name]       Save current session
  /history           Show conversation history
  /cost              Show token/cost summary
  /ctx               Show context items
  /tokens            Show estimated token count
  /compact           Force context compaction
  /todo              Show/manage todos
  /config            Show current configuration
  /providers         List configured providers
  /providers use <n> Switch to a provider profile
  /model [name]      Get/set model (shows history if no args)
  /maxctx [n]        Get/set max context tokens
  /iter [n]          Get/set max iterations
  /log [n]           Show last n lines of log (default 30)
  /bg                List background processes
  /undo              Undo last file change
  /redo              Redo last undo
  /mcp               Show MCP servers
  !<command>         Execute a shell command
`)
}

func (r *REPL) newSession(name string) {
	sess := NewSession(r.Workspace)
	if name != "" {
		sess.Name = name
	}
	r.Agent.SetSession(sess)

	if name != "" {
		r.SessionPath = getSessionPath(r.Workspace, name)
		r.Agent.SessionPath = r.SessionPath
	}
	fmt.Println("New session started.")
}

func (r *REPL) listSessions() {
	sessions := listAllSessions(r.Workspace)
	if len(sessions) == 0 {
		fmt.Println("No saved sessions.")
		return
	}

	sort.Strings(sessions)
	for _, name := range sessions {
		marker := "  "
		if r.Agent.Session != nil && r.Agent.Session.Name == name {
			marker = "▸ "
		}
		fmt.Printf("%s%s\n", marker, name)
	}
}

func (r *REPL) switchSession(name string) {
	if name == "" {
		fmt.Println("Usage: /session <name>")
		return
	}

	path := getSessionPath(r.Workspace, name)
	sess, err := LoadSession(path)
	if err != nil {
		fmt.Printf("Failed to load session '%s': %v\n", name, err)
		return
	}

	sess.Name = name
	r.Agent.SetSession(sess)
	r.SessionPath = path
	r.Agent.SessionPath = path

	fmt.Printf("Loaded session '%s' (%d messages)\n", name, len(sess.Messages))
}

func (r *REPL) deleteSession(name string) {
	if name == "" {
		fmt.Println("Usage: /delete <name>")
		return
	}

	path := getSessionPath(r.Workspace, name)
	if err := os.Remove(path); err != nil {
		fmt.Printf("Failed to delete session '%s': %v\n", name, err)
		return
	}
	fmt.Printf("Deleted session '%s'\n", name)
}

func (r *REPL) clearSession() {
	r.Agent.SetSession(NewSession(r.Workspace))
	fmt.Println("Session cleared.")
}

func (r *REPL) saveSession(name string) {
	if name != "" {
		r.Agent.Session.Name = name
		r.SessionPath = getSessionPath(r.Workspace, name)
		r.Agent.SessionPath = r.SessionPath
	}

	if r.SessionPath == "" {
		// Generate a name from the first user message
		firstMsg := r.Agent.Session.LastUserMessage()
		genName := GenerateSessionName(firstMsg)
		r.SessionPath = getSessionPath(r.Workspace, genName)
		r.Agent.SessionPath = r.SessionPath
		r.Agent.Session.Name = genName
	}

	if err := SaveSession(r.Agent.Session, r.SessionPath); err != nil {
		fmt.Printf("Save failed: %v\n", err)
		return
	}
	fmt.Printf("Session saved: %s\n", r.SessionPath)
}

func (r *REPL) showHistory() {
	if r.Agent.Session == nil || len(r.Agent.Session.Messages) == 0 {
		fmt.Println("No conversation history.")
		return
	}

	for i, m := range r.Agent.Session.Messages {
		content := MessageContent(m)
		preview := content
		if len(preview) > 100 {
			preview = preview[:100] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")
		fmt.Printf("[%d] %s: %s\n", i, m.Role, preview)
	}
}

func (r *REPL) showContext() {
	if r.Agent.Session == nil || len(r.Agent.Session.Context) == 0 {
		fmt.Println("No context items.")
		return
	}

	for _, item := range r.Agent.Session.Context {
		preview := item.Content
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")
		fmt.Printf("[%d] %s: %s — %s\n", item.ID, item.Type, item.Source, preview)
	}
}

func (r *REPL) showConfig() {
	fmt.Printf("Provider: %s\n", r.Config.Provider)
	fmt.Printf("Model:    %s\n", r.Config.Model)
	fmt.Printf("API Base: %s\n", r.Config.APIBase)
	fmt.Printf("Max Ctx:  %d\n", r.Config.MaxContextTokens)
	if r.Config.MaxTokens > 0 {
		fmt.Printf("Max Tok:  %d\n", r.Config.MaxTokens)
	}
}

func (r *REPL) showProviders() {
	if len(r.Config.Providers) == 0 {
		fmt.Println("No providers configured.")
		return
	}
	for name, p := range r.Config.Providers {
		marker := "  "
		if name == r.Config.Provider {
			marker = "▸ "
		}
		fmt.Printf("%s%-12s  model=%s  base=%s\n", marker, name, p.Model, p.APIURL)
	}
}

func (r *REPL) showMCP() {
	if len(r.Config.MCP) == 0 {
		fmt.Println("No MCP servers configured.")
		return
	}
	for name, m := range r.Config.MCP {
		fmt.Printf("  %-16s  type=%s  cmd=%s  enabled=%v\n",
			name,
			m["type"],
			m["command"],
			m["enabled"],
		)
	}
}

// handleProvidersSubcommand handles /providers use <name>, etc.
func (r *REPL) handleProvidersSubcommand(args string) {
	parts := strings.Fields(args)
	if len(parts) == 0 {
		r.showProviders()
		return
	}

	sub := strings.ToLower(parts[0])
	switch sub {
	case "use":
		if len(parts) < 2 {
			fmt.Println("Usage: /providers use <profile_name>")
			return
		}
		profile := parts[1]

		// Support numeric shorthand: /providers use 1
		if n, err := strconv.Atoi(profile); err == nil {
			names := make([]string, 0, len(r.Config.Providers))
			for name := range r.Config.Providers {
				names = append(names, name)
			}
			sort.Strings(names)
			if n >= 1 && n <= len(names) {
				profile = names[n-1]
			} else {
				fmt.Printf("No provider at index %d. Use /providers to see the list.\n", n)
				return
			}
		}

		p, ok := r.Config.Providers[profile]
		if !ok {
			fmt.Printf("Provider profile '%s' not found. Use /providers to see the list.\n", profile)
			return
		}

		// Apply provider settings to config
		if p.APIURL != "" {
			r.Config.APIBase = p.APIURL
		}
		if p.APIKey != "" {
			r.Config.APIKey = p.APIKey
		}
		if p.Model != "" {
			r.Config.Model = p.Model
		}
		if p.MaxTokens > 0 {
			r.Config.MaxTokens = p.MaxTokens
		}
		if p.Temperature > 0 {
			r.Config.Temperature = p.Temperature
		}
		r.Config.Provider = profile

		// Recreate streaming client with new settings
		r.Agent.Client = NewStreamingClient(r.Config)

		// Persist to ~/.z.json
		saveActiveConfigFields(map[string]interface{}{
			"api_url":     r.Config.APIBase,
			"api_key":     r.Config.APIKey,
			"model":       r.Config.Model,
			"max_tokens":  r.Config.MaxTokens,
			"temperature": r.Config.Temperature,
		})

		// Record to model history
		recordModelHistory(r.Config.Model, profile)

		detected := detectProviderLabel(r.Config.APIBase)
		fmt.Printf("✓ Switched to %s — %s / %s\n", profile, detected, r.Config.Model)

	default:
		fmt.Printf("Unknown /providers subcommand: %s\nUsage: /providers [use <name>]\n", sub)
	}
}

// showModelInfo shows current model and MRU history.
func (r *REPL) showModelInfo() {
	fmt.Printf("Current model: %s\n", r.Config.Model)
	fmt.Printf("Provider:      %s\n", r.Config.Provider)

	history := loadModelHistory()
	if len(history) > 0 {
		fmt.Println("\nRecent models:")
		for i, h := range history {
			if i >= 10 {
				break
			}
			marker := "  "
			if h.Model == r.Config.Model && h.Profile == r.Config.Provider {
				marker = "▸ "
			}
			fmt.Printf("%s%d. %s (%s)\n", marker, i+1, h.Model, h.Profile)
		}
		fmt.Println("\nUse /model <name> to switch or /model <n> to pick from history.")
	}
}

// handleModelSwitch handles /model <name> or /model <number>.
func (r *REPL) handleModelSwitch(args string) {
	// Check if it's a number from model history
	if n, err := strconv.Atoi(strings.TrimSpace(args)); err == nil {
		history := loadModelHistory()
		if n >= 1 && n <= len(history) {
			entry := history[n-1]
			// Switch to the provider profile associated with this model
			if p, ok := r.Config.Providers[entry.Profile]; ok {
				if p.APIURL != "" {
					r.Config.APIBase = p.APIURL
				}
				if p.APIKey != "" {
					r.Config.APIKey = p.APIKey
				}
				r.Config.Provider = entry.Profile
			}
			r.Config.Model = entry.Model
			r.Agent.Client = NewStreamingClient(r.Config)

			saveActiveConfigFields(map[string]interface{}{
				"api_url": r.Config.APIBase,
				"api_key": r.Config.APIKey,
				"model":   r.Config.Model,
			})
			recordModelHistory(r.Config.Model, r.Config.Provider)

			detected := detectProviderLabel(r.Config.APIBase)
			fmt.Printf("✓ Switched to %s via %s (%s)\n", r.Config.Model, r.Config.Provider, detected)
			return
		}
		fmt.Printf("No model at index %d in history.\n", n)
		return
	}

	// Direct model name set
	r.Config.Model = strings.TrimSpace(args)
	r.Agent.Client = NewStreamingClient(r.Config)

	saveActiveConfigFields(map[string]interface{}{
		"model": r.Config.Model,
	})
	recordModelHistory(r.Config.Model, r.Config.Provider)

	fmt.Printf("Model set to: %s\n", r.Config.Model)
}

// showLog shows the last n lines of the harness log file.
func (r *REPL) showLog(args string) {
	logFile := filepath.Join(r.Workspace, ".harness_output", "harness.log")
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		fmt.Println("No log file yet.")
		return
	}

	n := 30
	if args != "" {
		if parsed, err := strconv.Atoi(strings.TrimSpace(args)); err == nil && parsed > 0 {
			n = parsed
		}
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		fmt.Printf("Error reading log: %v\n", err)
		return
	}

	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	info, _ := os.Stat(logFile)
	sizeKB := float64(info.Size()) / 1024

	fmt.Printf("Log: %s (%.0f KB)\n", logFile, sizeKB)

	start := 0
	if len(lines) > n {
		start = len(lines) - n
	}
	for _, line := range lines[start:] {
		fmt.Printf("  %s\n", line)
	}
}

// handleIter shows or sets max iterations.
func (r *REPL) handleIter(args string) {
	if args == "" {
		fmt.Printf("Max iterations: %d\n", r.Agent.MaxIterations)
		return
	}

	n, err := strconv.Atoi(strings.TrimSpace(args))
	if err != nil {
		fmt.Println("Usage: /iter <number>")
		return
	}
	if n < 1 {
		fmt.Println("Must be at least 1")
		return
	}
	r.Agent.MaxIterations = n
	fmt.Printf("✓ Max iterations set to %d\n", n)
}

// showBackgroundProcesses lists running background processes.
func (r *REPL) showBackgroundProcesses() {
	r.Agent.Tools.mu.Lock()
	defer r.Agent.Tools.mu.Unlock()

	if len(r.Agent.Tools.bgProcs) == 0 {
		fmt.Println("No background processes.")
		return
	}

	for _, p := range r.Agent.Tools.bgProcs {
		status := "running"
		if p.Done {
			status = fmt.Sprintf("exited (code %d)", p.ExitCode)
		}
		elapsed := time.Since(p.StartTime).Truncate(time.Second)
		fmt.Printf("  [%d] PID=%d  %s  %s  (%s)\n", p.ID, p.PID, status, p.Command, elapsed)
	}
}

// RunPiped handles piped (non-interactive) input.
func RunPiped(agent *Agent) {
	// Read all stdin
	data := make([]byte, 0)
	buf := make([]byte, 4096)
	for {
		n, err := os.Stdin.Read(buf)
		if n > 0 {
			data = append(data, buf[:n]...)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			return
		}
	}

	input := strings.TrimSpace(string(data))
	if input == "" {
		fmt.Fprintln(os.Stderr, "No input received from pipe.")
		return
	}

	_ = agent.RunMessage(input) // Already displayed by the agent
}
