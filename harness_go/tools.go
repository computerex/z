package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ToolResult struct {
	Success bool
	Message string
	Content string
}

type ParsedToolCall struct {
	Name       string
	Parameters map[string]string
}

type ToolRegistry struct {
	workspace string
}

func NewToolRegistry(workspace string) *ToolRegistry {
	return &ToolRegistry{workspace: workspace}
}

func (tr *ToolRegistry) resolvePath(path string) string {
	p := filepath.Clean(path)
	if !filepath.IsAbs(path) {
		p = filepath.Join(tr.workspace, p)
	}
	return filepath.Clean(p)
}

func (tr *ToolRegistry) ExecuteTool(tool *ParsedToolCall) *ToolResult {
	switch tool.Name {
	case "read_file":
		return tr.readFile(tool.Parameters)
	case "write_to_file":
		return tr.writeFile(tool.Parameters)
	case "replace_in_file":
		return tr.replaceInFile(tool.Parameters)
	case "execute_command":
		return tr.executeCommand(tool.Parameters)
	case "list_files":
		return tr.listFiles(tool.Parameters)
	case "search_files":
		return tr.searchFiles(tool.Parameters)
	case "check_background_process":
		return tr.checkBackgroundProcess(tool.Parameters)
	case "stop_background_process":
		return tr.stopBackgroundProcess(tool.Parameters)
	case "list_background_processes":
		return tr.listBackgroundProcesses(tool.Parameters)
	case "list_context":
		return tr.listContext()
	case "remove_from_context":
		return tr.removeFromContext(tool.Parameters)
	case "attempt_completion":
		return tr.attemptCompletion(tool.Parameters)
	default:
		return &ToolResult{Success: false, Message: fmt.Sprintf("Unknown tool: %s", tool.Name)}
	}
}

func (tr *ToolRegistry) readFile(params map[string]string) *ToolResult {
	path := tr.resolvePath(params["path"])

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: File not found: %s", path)}
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error reading file: %v", err)}
	}

	text := string(content)
	lines := strings.Split(text, "\n")
	numbered := make([]string, len(lines))
	for i, line := range lines {
		numbered[i] = fmt.Sprintf("%4d | %s", i+1, line)
	}
	result := strings.Join(numbered, "\n")

	return &ToolResult{
		Success: true,
		Content: result,
		Message: fmt.Sprintf("Read %d lines from %s", len(lines), path),
	}
}

func (tr *ToolRegistry) writeFile(params map[string]string) *ToolResult {
	path := tr.resolvePath(params["path"])
	content := params["content"]

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error creating directory: %v", err)}
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error writing file: %v", err)}
	}

	return &ToolResult{
		Success: true,
		Message: fmt.Sprintf("Successfully wrote to %s", path),
	}
}

func (tr *ToolRegistry) replaceInFile(params map[string]string) *ToolResult {
	path := tr.resolvePath(params["path"])
	diff := params["diff"]

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: File not found: %s", path)}
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error reading file: %v", err)}
	}

	blocks := parseSearchReplaceBlocks(diff)
	if len(blocks) == 0 {
		return &ToolResult{Success: false, Message: "Error: No valid SEARCH/REPLACE blocks found"}
	}

	newContent := string(content)
	changes := 0
	for _, block := range blocks {
		search, replace := block[0], block[1]
		if strings.Contains(newContent, search) {
			newContent = strings.Replace(newContent, search, replace, 1)
			changes++
		} else {
			searchPreview := search
			if len(searchPreview) > 200 {
				searchPreview = searchPreview[:200]
			}
			return &ToolResult{Success: false, Message: fmt.Sprintf("Error: SEARCH block not found in file:\n%s...", searchPreview)}
		}
	}

	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error writing file: %v", err)}
	}

	return &ToolResult{
		Success: true,
		Message: fmt.Sprintf("Successfully made %d replacement(s) in %s", changes, path),
	}
}

func (tr *ToolRegistry) executeCommand(params map[string]string) *ToolResult {
	command := params["command"]
	background := params["background"] == "true"
	timeoutSecs := 120

	fmt.Printf("\n$ %s\n", command)

	if background {
		return tr.runBackgroundCommand(command)
	}

	isWindows := runtime.GOOS == "windows"

	var cmd *exec.Cmd
	if isWindows {
		cmd = exec.Command("cmd.exe", "/c", command)
	} else {
		cmd = exec.Command("bash", "-c", command)
	}

	cmd.Dir = tr.workspace

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: %v", err)}
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: %v", err)}
	}

	if err := cmd.Start(); err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error starting command: %v", err)}
	}

	outputLines := []string{}

	reader := io.MultiReader(stdout, stderr)
	scanner := bufio.NewScanner(reader)
	done := make(chan bool)

	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			outputLines = append(outputLines, line)
			fmt.Printf("  %s\n", line)
		}
		done <- true
	}()

	timeout := time.Duration(timeoutSecs) * time.Second
	select {
	case <-done:
		cmd.Wait()
	case <-time.After(timeout):
		cmd.Process.Kill()
		return &ToolResult{
			Success: true,
			Message: fmt.Sprintf("Command timed out after %ds", timeoutSecs),
		}
	}

	exitCode := 0
	if cmd.ProcessState != nil {
		exitCode = cmd.ProcessState.ExitCode()
	}

	if exitCode == 0 {
		fmt.Printf("[OK] Exit code: %d\n", exitCode)
	} else {
		fmt.Printf("[X] Exit code: %d\n", exitCode)
	}

	output := strings.Join(outputLines, "\n")
	if output == "" {
		output = "(no output)"
	}

	return &ToolResult{
		Success: exitCode == 0,
		Message: output,
		Content: output,
	}
}

func (tr *ToolRegistry) runBackgroundCommand(command string) *ToolResult {
	bgID := generateBgID()
	isWindows := runtime.GOOS == "windows"

	var cmd *exec.Cmd
	if isWindows {
		cmd = exec.Command("cmd.exe", "/c", command)
	} else {
		cmd = exec.Command("bash", "-c", command)
	}

	cmd.Dir = tr.workspace

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: %v", err)}
	}

	if err := cmd.Start(); err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error starting command: %v", err)}
	}

	outputLines := []string{}
	scanner := bufio.NewScanner(stdout)
	done := make(chan bool)

	go func() {
		count := 0
		for scanner.Scan() && count < 10 {
			line := scanner.Text()
			outputLines = append(outputLines, line)
			fmt.Printf("  %s\n", line)
			count++
		}
		done <- true
	}()

	<-done

	bgProcsMu.Lock()
	bgProcs[bgID] = &BackgroundProcess{
		ID:      bgID,
		Command: command,
		Proc:    cmd,
		Started: time.Now(),
		Logs:    outputLines,
	}
	bgProcsMu.Unlock()

	fmt.Printf("[green]-> Running in background (ID: %d, PID: %d)[/green]\n", bgID, cmd.Process.Pid)

	return &ToolResult{
		Success: true,
		Message: fmt.Sprintf("Command started in background (ID: %d, PID: %d)", bgID, cmd.Process.Pid),
	}
}

func (tr *ToolRegistry) listFiles(params map[string]string) *ToolResult {
	pathStr := params["path"]
	if pathStr == "" {
		pathStr = "."
	}
	path := tr.resolvePath(pathStr)

	recursive := params["recursive"] == "true"

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: Directory not found: %s", path)}
	}

	items := []string{}
	maxItems := 50
	if recursive {
		maxItems = 100
	}

	// User explicitly requested a hidden directory if path starts with . (but not just ".")
	userRequestedHidden := strings.HasPrefix(pathStr, ".") && pathStr != "."

	skipDirs := map[string]bool{
		"node_modules": true, "__pycache__": true, "venv": true,
		"dist": true, "build": true, "target": true, "vendor": true, "obj": true, "bin": true,
	}

	// shouldSkip returns true if we should skip this directory name
	shouldSkip := func(name string) bool {
		if userRequestedHidden {
			return false
		}
		// Skip dotfiles/dotdirs
		if strings.HasPrefix(name, ".") {
			return true
		}
		return skipDirs[name]
	}

	truncated := false

	var walkFn func(string, int) bool
	walkFn = func(dir string, depth int) bool {
		if len(items) >= maxItems {
			truncated = true
			return false
		}

		entries, err := os.ReadDir(dir)
		if err != nil {
			return true
		}

		for _, entry := range entries {
			if len(items) >= maxItems {
				truncated = true
				return false
			}

			fullPath := filepath.Join(dir, entry.Name())
			relPath, err := filepath.Rel(path, fullPath)
			if err != nil {
				continue
			}

			parts := strings.Split(filepath.ToSlash(relPath), "/")
			skip := false
			for _, part := range parts {
				if shouldSkip(part) {
					skip = true
					break
				}
			}
			if skip {
				continue
			}

			if entry.IsDir() {
				if recursive {
					items = append(items, relPath+"/")
					if !walkFn(fullPath, depth+1) {
						return false
					}
				}
			} else {
				items = append(items, relPath)
			}
		}
		return true
	}

	if recursive {
		walkFn(path, 0)
	} else {
		entries, _ := os.ReadDir(path)
		for _, entry := range entries {
			if len(items) >= maxItems {
				truncated = true
				break
			}

			relPath := entry.Name()

			if shouldSkip(relPath) {
				continue
			}

			if entry.IsDir() {
				items = append(items, relPath+"/")
			} else {
				items = append(items, relPath)
			}
		}
	}

	result := strings.Join(items, "\n")
	if result == "" {
		result = "(empty directory)"
	}
	if truncated {
		result += fmt.Sprintf("\n\n... (truncated at %d items)", maxItems)
	}

	return &ToolResult{
		Success: true,
		Content: result,
		Message: result,
	}
}

// errMaxResults is returned to stop filepath.Walk when we have enough results
var errMaxResults = fmt.Errorf("max results reached")

func (tr *ToolRegistry) searchFiles(params map[string]string) *ToolResult {
	pathStr := params["path"]
	if pathStr == "" {
		pathStr = "."
	}
	path := tr.resolvePath(pathStr)

	regex := params["regex"]
	filePattern := params["file_pattern"]
	if filePattern == "" {
		filePattern = "*"
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: Directory not found: %s", path)}
	}

	pattern, err := regexp.Compile("(?i)" + regex)
	if err != nil {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: Invalid regex: %v", err)}
	}

	// User explicitly requested a hidden directory if path starts with . (but not just ".")
	userRequestedHidden := strings.HasPrefix(pathStr, ".") && pathStr != "."

	// Directories to skip (non-dotfile junk)
	skipDirs := map[string]bool{
		"node_modules": true, "__pycache__": true, "venv": true,
		"dist": true, "build": true, "target": true, "vendor": true, "bin": true, "obj": true,
	}

	// shouldSkipDir returns true if we should skip this directory
	shouldSkipDir := func(name string) bool {
		if userRequestedHidden {
			return false
		}
		// Skip any directory starting with .
		if strings.HasPrefix(name, ".") {
			return true
		}
		return skipDirs[name]
	}

	results := []string{}
	maxResults := 100
	filesScanned := 0
	maxFilesToScan := 5000 // Safety limit

	filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		// Skip common large directories and dotdirs
		if info.IsDir() {
			if shouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}

		// Stop if we have enough results
		if len(results) >= maxResults {
			return errMaxResults
		}

		// Safety limit on files scanned
		filesScanned++
		if filesScanned >= maxFilesToScan {
			return errMaxResults
		}

		matched, _ := filepath.Match(filePattern, filepath.Base(filePath))
		if !matched {
			return nil
		}

		// Skip large files (>1MB) and binary files
		if info.Size() > 1024*1024 {
			return nil
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil
		}

		// Skip binary files (check for null bytes in first 512 bytes)
		checkLen := 512
		if len(content) < checkLen {
			checkLen = len(content)
		}
		for i := 0; i < checkLen; i++ {
			if content[i] == 0 {
				return nil // Binary file
			}
		}

		relPath, _ := filepath.Rel(path, filePath)
		lines := strings.Split(string(content), "\n")
		for i, line := range lines {
			if pattern.MatchString(line) {
				linePreview := line
				if len(linePreview) > 150 {
					linePreview = linePreview[:150]
				}
				results = append(results, fmt.Sprintf("%s:%d: %s", relPath, i+1, linePreview))
				if len(results) >= maxResults {
					return errMaxResults
				}
			}
		}

		return nil
	})

	if len(results) == 0 {
		return &ToolResult{
			Success: true,
			Message: "(no matches)",
		}
	}

	result := strings.Join(results, "\n")
	return &ToolResult{
		Success: true,
		Content: result,
		Message: result,
	}
}

type BackgroundProcess struct {
	ID        int
	Command   string
	Proc      *exec.Cmd
	Started   time.Time
	Logs      []string
	LastCheck int
}

var (
	bgProcs  = make(map[int]*BackgroundProcess)
	bgProcsMu sync.Mutex
	nextBgID  = 1
)

func generateBgID() int {
	bgProcsMu.Lock()
	defer bgProcsMu.Unlock()
	id := nextBgID
	nextBgID++
	return id
}

func (tr *ToolRegistry) checkBackgroundProcess(params map[string]string) *ToolResult {
	idStr := params["id"]
	lines := 50

	if idStr != "" {
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return &ToolResult{Success: false, Message: "Error: ID must be a number"}
		}

		bgProcsMu.Lock()
		proc, ok := bgProcs[id]
		bgProcsMu.Unlock()

		if !ok {
			return &ToolResult{Success: false, Message: fmt.Sprintf("Error: No background process with ID %d", id)}
		}

		elapsed := time.Since(proc.Started).Seconds()
		status := "running"
		if proc.Proc.ProcessState != nil {
			status = fmt.Sprintf("exited (%d)", proc.Proc.ProcessState.ExitCode())
		}

		recentLogs := proc.Logs
		if len(recentLogs) > lines {
			recentLogs = recentLogs[len(recentLogs)-lines:]
		}

		result := fmt.Sprintf("Background Process [%d]\n", id)
		result += fmt.Sprintf("Command: %s\n", proc.Command)
		result += fmt.Sprintf("PID: %d\n", proc.Proc.Process.Pid)
		result += fmt.Sprintf("Status: %s\n", status)
		result += fmt.Sprintf("Running time: %.0fs\n", elapsed)
		result += fmt.Sprintf("Total log lines: %d\n", len(proc.Logs))
		result += fmt.Sprintf("\n--- Last %d lines ---\n", len(recentLogs))
		result += strings.Join(recentLogs, "\n")

		if proc.Proc.ProcessState == nil {
			result += "\n\n[!] Process still running. Continue with other tasks."
		}

		return &ToolResult{
			Success: true,
			Message: result,
		}
	}

	bgProcsMu.Lock()
	defer bgProcsMu.Unlock()

	if len(bgProcs) == 0 {
		return &ToolResult{Success: true, Message: "No background processes running."}
	}

	result := "Background processes:\n"
	for id, proc := range bgProcs {
		elapsed := time.Since(proc.Started).Minutes()
		status := "running"
		if proc.Proc.ProcessState != nil {
			status = fmt.Sprintf("exited (%d)", proc.Proc.ProcessState.ExitCode())
		}
		result += fmt.Sprintf("  [%d] PID %d - %s - %.1fm - %s\n", id, proc.Proc.Process.Pid, status, elapsed, proc.Command)
	}

	return &ToolResult{
		Success: true,
		Message: result,
	}
}

func (tr *ToolRegistry) stopBackgroundProcess(params map[string]string) *ToolResult {
	idStr := params["id"]
	id, err := strconv.Atoi(idStr)
	if err != nil {
		return &ToolResult{Success: false, Message: "Error: ID must be a number"}
	}

	bgProcsMu.Lock()
	proc, ok := bgProcs[id]
	bgProcsMu.Unlock()

	if !ok {
		return &ToolResult{Success: false, Message: fmt.Sprintf("Error: No background process with ID %d", id)}
	}

	if proc.Proc.ProcessState != nil {
		return &ToolResult{Success: true, Message: fmt.Sprintf("Process [%d] already exited with code %d", id, proc.Proc.ProcessState.ExitCode())}
	}

	if runtime.GOOS == "windows" {
		cmd := exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprint(proc.Proc.Process.Pid))
		cmd.Run()
	} else {
		if proc.Proc.Process != nil {
			proc.Proc.Process.Kill()
		}
	}

	bgProcsMu.Lock()
	delete(bgProcs, id)
	bgProcsMu.Unlock()

	return &ToolResult{
		Success: true,
		Message: fmt.Sprintf("Stopped background process [%d] (PID: %d)", id, proc.Proc.Process.Pid),
	}
}

func (tr *ToolRegistry) listBackgroundProcesses(params map[string]string) *ToolResult {
	bgProcsMu.Lock()
	defer bgProcsMu.Unlock()

	if len(bgProcs) == 0 {
		return &ToolResult{Success: true, Message: "No background processes."}
	}

	result := "Background processes:\n"
	for id, proc := range bgProcs {
		elapsed := time.Since(proc.Started).Minutes()
		status := "running"
		if proc.Proc.ProcessState != nil {
			status = fmt.Sprintf("exited (%d)", proc.Proc.ProcessState.ExitCode())
		}
		result += fmt.Sprintf("  [%d] PID %d - %s - %.1fm - %s\n", id, proc.Proc.Process.Pid, status, elapsed, proc.Command)
	}

	return &ToolResult{
		Success: true,
		Message: result,
	}
}

func (tr *ToolRegistry) listContext() *ToolResult {
	return &ToolResult{Success: true, Message: "Context listing not yet implemented in this version"}
}

func (tr *ToolRegistry) removeFromContext(params map[string]string) *ToolResult {
	return &ToolResult{Success: true, Message: "Context removal not yet implemented in this version"}
}

func (tr *ToolRegistry) attemptCompletion(params map[string]string) *ToolResult {
	return &ToolResult{
		Success: true,
		Message: "Task completed.",
	}
}

func ParseXMLTool(content string) *ParsedToolCall {
	toolNames := []string{
		"read_file", "write_to_file", "replace_in_file",
		"execute_command", "list_files", "search_files",
		"check_background_process", "stop_background_process", "list_background_processes",
		"list_context", "remove_from_context", "analyze_image", "web_search",
		"attempt_completion",
	}

	for _, toolName := range toolNames {
		// Use (?s) flag to make . match newlines
		pattern := regexp.MustCompile(fmt.Sprintf(`(?s)<%s>(.*?)</%s>`, toolName, toolName))
		match := pattern.FindStringSubmatch(content)
		if match != nil {
			inner := match[1]
			params := parseParams(inner)
			return &ParsedToolCall{
				Name:       toolName,
				Parameters: params,
			}
		}
	}

	return nil
}

func parseParams(inner string) map[string]string {
	params := make(map[string]string)

	// Known parameter names
	paramNames := []string{"path", "content", "diff", "command", "background", 
		"recursive", "regex", "file_pattern", "result", "id", "lines", "question", "query", "count", "source"}

	for _, paramName := range paramNames {
		// Use (?s) to make . match newlines
		pattern := regexp.MustCompile(fmt.Sprintf(`(?s)<%s>(.*?)</%s>`, paramName, paramName))
		match := pattern.FindStringSubmatch(inner)
		if match != nil && len(match) >= 2 {
			value := strings.Trim(match[1], "\n")
			params[paramName] = value
		}
	}

	return params
}

func parseSearchReplaceBlocks(diff string) [][]string {
	pattern := regexp.MustCompile(`<<<<<<<\s*SEARCH\n(.*?)\n={7}\n(.*?)\n>{7}\s*REPLACE`)
	matches := pattern.FindAllStringSubmatch(diff, -1)

	blocks := make([][]string, len(matches))
	for i, m := range matches {
		blocks[i] = []string{m[1], m[2]}
	}

	return blocks
}