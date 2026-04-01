package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// AgentMeta is persisted to ~/COS/agents/<id>/meta.json
type AgentMeta struct {
	Task      string  `json:"task"`
	Status    string  `json:"status"` // running, done, error, killed
	Started   string  `json:"started"`
	Worktree  string  `json:"worktree"`
	PID       int     `json:"pid"`
	Model     string  `json:"model,omitempty"`
	SessionID string  `json:"session_id,omitempty"`
	CostUSD   float64 `json:"cost_usd,omitempty"`
	NumTurns  int     `json:"num_turns,omitempty"`
	Elapsed   float64 `json:"elapsed_seconds,omitempty"`
	ExitCode  int     `json:"exit_code,omitempty"`
	Errors    []string `json:"errors,omitempty"`
}

func agentDir(id string) string          { return filepath.Join(agentsDir, id) }
func agentMetaPath(id string) string     { return filepath.Join(agentDir(id), "meta.json") }
func agentLogPath(id string) string      { return filepath.Join(agentDir(id), "output.log") }
func agentStderrPath(id string) string   { return filepath.Join(agentDir(id), "stderr.log") }
func agentWrapperPath(id string) string  { return filepath.Join(agentDir(id), "_run.go") }

func loadAgentMeta(id string) (*AgentMeta, error) {
	data, err := os.ReadFile(agentMetaPath(id))
	if err != nil {
		return nil, err
	}
	var m AgentMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func saveAgentMeta(id string, m *AgentMeta) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(agentMetaPath(id), data, 0644)
}

// pidAlive checks if a process is still running.
// Implementation is in kill_unix.go and kill_windows.go.

// selfExe returns the path to the current cos binary.
func selfExe() string {
	exe, err := os.Executable()
	if err != nil {
		return "cos"
	}
	return exe
}

// findZ locates the z (harness) executable on PATH.
func findZ() string {
	path, err := exec.LookPath("z")
	if err == nil {
		return path
	}
	// Fallback: try python -m harness.cli
	pyPath, err := exec.LookPath("python3")
	if err != nil {
		pyPath, _ = exec.LookPath("python")
	}
	if pyPath != "" {
		return pyPath + " -m harness.cli"
	}
	return "z"
}

// spawnAgent launches a detached z subprocess for the given prompt.
func spawnAgent(id, prompt, task, worktree, model string) (int, error) {
	dir := agentDir(id)
	os.MkdirAll(dir, 0755)

	cfg := loadConfig()
	if model == "" {
		model = cfg.SubAgentModel
		if model == "" {
			model = cfg.Model
		}
	}

	meta := &AgentMeta{
		Task:     task,
		Status:   "running",
		Started:  nowISO(),
		Worktree: worktree,
		Model:    model,
	}
	if task == "" && len(prompt) > 80 {
		meta.Task = prompt[:80]
	} else if task == "" {
		meta.Task = prompt
	}
	saveAgentMeta(id, meta)

	// Build a small Go wrapper that runs z, captures output, and notifies.
	// We compile and run it detached.
	// Actually — simpler: write a helper script and run it.
	// Even simpler: just spawn `z` directly with piped stdin.
	zExe := findZ()
	logPath := agentLogPath(id)
	stderrPath := agentStderrPath(id)

	// Build z args
	args := buildZArgs(zExe, worktree, id, model)

	cwd := worktree
	if cwd == "" || !dirExists(cwd) {
		cwd = cosDir
	}

	// Open log files
	logF, err := os.Create(logPath)
	if err != nil {
		return 0, fmt.Errorf("create log: %w", err)
	}
	stderrF, err := os.Create(stderrPath)
	if err != nil {
		logF.Close()
		return 0, fmt.Errorf("create stderr log: %w", err)
	}

	cmd := buildCommand(args, cwd)
	cmd.Stdin = strings.NewReader(prompt)
	cmd.Stdout = logF
	cmd.Stderr = stderrF
	detachProcess(cmd)

	if err := cmd.Start(); err != nil {
		logF.Close()
		stderrF.Close()
		return 0, fmt.Errorf("start: %w", err)
	}

	pid := cmd.Process.Pid
	meta.PID = pid
	saveAgentMeta(id, meta)

	// Close log files — the child process holds its own handles.
	logF.Close()
	stderrF.Close()

	// Spawn a detached reaper process that waits for the child, updates
	// meta.json and writes the notification file. This survives even if
	// the parent cos process exits immediately.
	reaperArgs := []string{selfExe(), "agent", "_wait", id, fmt.Sprintf("%d", pid)}
	reaperCmd := exec.Command(reaperArgs[0], reaperArgs[1:]...)
	reaperCmd.Dir = cosDir
	reaperCmd.Stdout = nil
	reaperCmd.Stderr = nil
	reaperCmd.Stdin = nil
	detachProcess(reaperCmd)
	if err := reaperCmd.Start(); err != nil {
		// Reaper failed to start — not fatal, status will be fixed lazily
		fmt.Fprintf(os.Stderr, "Warning: reaper start failed: %v\n", err)
	}

	return pid, nil
}

func buildZArgs(zExe, worktree, session, model string) []string {
	parts := strings.Fields(zExe)
	if worktree != "" && dirExists(worktree) {
		parts = append(parts, worktree)
	}
	parts = append(parts, "--session", session)
	if model != "" {
		parts = append(parts, "--model", model)
	}
	return parts
}

func buildCommand(args []string, cwd string) *exec.Cmd {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = cwd
	return cmd
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func parseTime(iso string) time.Time {
	t, err := time.Parse(time.RFC3339, iso)
	if err != nil {
		return time.Now()
	}
	return t
}

// waitForAgent is the reaper logic: waits for pid to exit, updates meta, notifies.
// Invoked by: cos agent _wait <id> <pid>
func waitForAgent(id string, pid int) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return
	}
	// Wait for the process (blocks until exit)
	state, _ := proc.Wait()

	meta, _ := loadAgentMeta(id)
	if meta == nil {
		return
	}
	if meta.Status == "killed" {
		return // don't overwrite a manual kill
	}

	elapsed := time.Since(parseTime(meta.Started)).Seconds()
	status := "done"
	exitCode := 0
	if state != nil && !state.Success() {
		status = "error"
		exitCode = state.ExitCode()
	}
	meta.Status = status
	meta.Elapsed = elapsed
	meta.ExitCode = exitCode
	saveAgentMeta(id, meta)

	// Touch notification file
	notifyPath := filepath.Join(notifyDir, id)
	os.WriteFile(notifyPath, []byte{}, 0644)
}

// resumeAgent sends a follow-up to an existing agent's session.
func resumeAgent(id, prompt string) (int, error) {
	meta, err := loadAgentMeta(id)
	if err != nil {
		return 0, fmt.Errorf("agent '%s' not found", id)
	}
	followupID := fmt.Sprintf("%s-followup-%d", id, time.Now().Unix())
	return spawnAgent(followupID, prompt, "Follow-up on "+id, meta.Worktree, meta.Model)
}

// --- CLI command handlers ---

func cmdListAgents() {
	entries, err := os.ReadDir(agentsDir)
	if err != nil || len(entries) == 0 {
		fmt.Println("No agents yet.")
		return
	}

	// Collect valid agent dirs
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			if _, err := os.Stat(agentMetaPath(e.Name())); err == nil {
				names = append(names, e.Name())
			}
		}
	}
	sort.Strings(names)

	if len(names) == 0 {
		fmt.Println("No agents yet.")
		return
	}

	icons := map[string]string{
		"running": "▶",
		"done":    "✅",
		"killed":  "✖",
		"error":   "❌",
	}

	for _, name := range names {
		meta, err := loadAgentMeta(name)
		if err != nil {
			continue
		}
		status := meta.Status
		if status == "running" && meta.PID > 0 && !pidAlive(meta.PID) {
			status = "done"
			// Persist the status fix so it sticks
			meta.Status = status
			saveAgentMeta(name, meta)
		}
		icon := icons[status]
		if icon == "" {
			icon = "○"
		}
		task := meta.Task
		if len(task) > 80 {
			task = task[:80]
		}
		fmt.Printf("%s %s  %s\n", icon, name, task)

		// Show last line of log
		logPath := agentLogPath(name)
		if data, err := os.ReadFile(logPath); err == nil {
			lines := nonEmptyLines(string(data))
			if len(lines) > 0 {
				last := lines[len(lines)-1]
				if len(last) > 100 {
					last = last[:100]
				}
				fmt.Printf("  └ %s\n", last)
			}
		}
	}
}

func cmdShowLog(id string, tail int) {
	logPath := agentLogPath(id)
	data, err := os.ReadFile(logPath)
	if err != nil {
		fmt.Printf("No log for agent '%s'\n", id)
		return
	}
	text := string(data)
	if tail > 0 {
		lines := strings.Split(text, "\n")
		if len(lines) > tail {
			lines = lines[len(lines)-tail:]
		}
		text = strings.Join(lines, "\n")
	}
	fmt.Print(text)
}

func cmdKillAgent(id string) {
	meta, err := loadAgentMeta(id)
	if err != nil {
		fmt.Printf("Agent '%s' not found.\n", id)
		return
	}
	if meta.PID > 0 && pidAlive(meta.PID) {
		killPID(meta.PID)
		meta.Status = "killed"
		saveAgentMeta(id, meta)
		fmt.Printf("Killed agent '%s' (PID %d).\n", id, meta.PID)
	} else {
		fmt.Printf("Agent '%s' not running.\n", id)
	}
}

func nonEmptyLines(s string) []string {
	var result []string
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}
