package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ToolResult holds the output of a tool execution.
type ToolResult struct {
	Output string
	Error  string
}

// Think token sanitization constants.
// ZWS (\u200b) is inserted between < and think> to prevent the XML parser
// from confusing model-generated <think> tags with real ones.
const thinkZWS = "\u200b"

// desanitizeThinkTokens reverses ZWS escaping in think tags.
func desanitizeThinkTokens(text string) string {
	text = strings.ReplaceAll(text, "<"+thinkZWS+"/think>", "</think>")
	text = strings.ReplaceAll(text, "<"+thinkZWS+"think>", "<think>")
	return text
}

// ToolHandlers executes tools in the workspace context.
type ToolHandlers struct {
	Workspace   string
	mu          sync.Mutex
	bgProcs     map[int]*bgProcess
	nextBgID    int
	spillDir    string
	spillCount  int
}

type bgProcess struct {
	ID        int
	PID       int
	Command   string
	StartTime time.Time
	Done      bool
	ExitCode  int
	LogPath   string
}

// NewToolHandlers creates a new tool handler set.
func NewToolHandlers(workspace string) *ToolHandlers {
	spillDir := filepath.Join(workspace, ".harness_output")
	os.MkdirAll(spillDir, 0755)
	return &ToolHandlers{
		Workspace: workspace,
		bgProcs:   make(map[int]*bgProcess),
		nextBgID:  1,
		spillDir:  spillDir,
	}
}

// Dispatch routes a tool call to the appropriate handler.
func (th *ToolHandlers) Dispatch(call ToolCall) ToolResult {
	logInfo("Tool dispatch: %s", call.Name)
	switch call.Name {
	case "read_file":
		return th.readFile(call.Params)
	case "write_to_file":
		return th.writeToFile(call.Params)
	case "replace_in_file":
		return th.replaceInFile(call.Params)
	case "replace_between_anchors":
		return th.replaceBetweenAnchors(call.Params)
	case "execute_command":
		return th.executeCommand(call.Params)
	case "list_files":
		return th.listFiles(call.Params)
	case "search_files":
		return th.searchFiles(call.Params)
	case "list_background_processes":
		return th.listBgProcs()
	case "check_background_process":
		return th.checkBgProc(call.Params)
	case "stop_background_process":
		return th.stopBgProc(call.Params)
	case "manage_todos":
		return ToolResult{Output: "TODO: manage_todos not yet implemented in Go port"}
	case "web_search":
		return ToolResult{Output: "Web search not available in Go port"}
	case "analyze_image":
		return ToolResult{Output: "Image analysis not available in Go port"}
	case "mcp_search_tools", "mcp_list_tools", "mcp_call_tool":
		return ToolResult{Output: "MCP tools not yet implemented in Go port"}
	case "retrieve_tool_result":
		return th.retrieveToolResult(call.Params)
	case "introspect":
		return ToolResult{Output: "(introspection complete)"}
	case "attempt_completion":
		return th.attemptCompletion(call.Params)
	default:
		return ToolResult{Error: fmt.Sprintf("Unknown tool: %s", call.Name)}
	}
}

func (th *ToolHandlers) absPath(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(th.Workspace, p)
}

// safePath resolves a path and validates it stays within the workspace.
// Returns the resolved absolute path and an error if it escapes.
func (th *ToolHandlers) safePath(p string) (string, error) {
	abs := th.absPath(p)
	// Clean to resolve ".." etc.
	abs = filepath.Clean(abs)
	ws := filepath.Clean(th.Workspace)

	// Check that the resolved path is under the workspace
	if !strings.HasPrefix(abs, ws+string(filepath.Separator)) && abs != ws {
		return "", fmt.Errorf("path %q escapes workspace %q", p, ws)
	}
	return abs, nil
}

// ── read_file ───────────────────────────────────────────────────────────

func (th *ToolHandlers) readFile(params map[string]string) ToolResult {
	path := params["path"]
	if path == "" {
		return ToolResult{Error: "read_file: missing required parameter 'path'"}
	}

	absP, err := th.safePath(path)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("read_file: %v", err)}
	}
	data, err := os.ReadFile(absP)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("read_file: %v", err)}
	}

	lines := strings.Split(string(data), "\n")
	startLine := 1
	endLine := len(lines)

	if s, ok := params["start_line"]; ok {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && n > 0 {
			startLine = n
		}
	}
	if e, ok := params["end_line"]; ok {
		if n, err := strconv.Atoi(strings.TrimSpace(e)); err == nil && n > 0 {
			endLine = n
		}
	}

	// Clamp
	if startLine < 1 {
		startLine = 1
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	if startLine > endLine {
		return ToolResult{Error: fmt.Sprintf("read_file: start_line %d > end_line %d", startLine, endLine)}
	}

	// For large files without explicit range, limit to first 300 lines
	if endLine-startLine > 300 && params["start_line"] == "" && params["end_line"] == "" {
		endLine = startLine + 299
		if endLine > len(lines) {
			endLine = len(lines)
		}
	}

	var sb strings.Builder
	for i := startLine - 1; i < endLine; i++ {
		sb.WriteString(fmt.Sprintf("%d | %s\n", i+1, lines[i]))
	}

	if endLine < len(lines) && params["end_line"] == "" {
		sb.WriteString(fmt.Sprintf("\n... (%d lines total, showing %d-%d)\n", len(lines), startLine, endLine))
	}

	return ToolResult{Output: sb.String()}
}

// ── write_to_file ───────────────────────────────────────────────────────

func (th *ToolHandlers) writeToFile(params map[string]string) ToolResult {
	path := params["path"]
	content := desanitizeThinkTokens(params["content"])
	if path == "" {
		return ToolResult{Error: "write_to_file: missing required parameter 'path'"}
	}

	absP, err := th.safePath(path)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("write_to_file: %v", err)}
	}

	// Create parent directories
	dir := filepath.Dir(absP)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return ToolResult{Error: fmt.Sprintf("write_to_file: cannot create directory: %v", err)}
	}

	// Strip leading/trailing newline from content (matches Python behavior)
	content = strings.TrimPrefix(content, "\n")
	content = strings.TrimSuffix(content, "\n")
	// Ensure file ends with newline
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}

	if err := os.WriteFile(absP, []byte(content), 0644); err != nil {
		return ToolResult{Error: fmt.Sprintf("write_to_file: %v", err)}
	}

	lineCount := strings.Count(content, "\n")
	return ToolResult{Output: fmt.Sprintf("File written: %s (%d lines)", path, lineCount)}
}

// ── replace_in_file ─────────────────────────────────────────────────────

func (th *ToolHandlers) replaceInFile(params map[string]string) ToolResult {
	path := params["path"]
	oldText := desanitizeThinkTokens(params["old_text"])
	newText := desanitizeThinkTokens(params["new_text"])
	if path == "" {
		return ToolResult{Error: "replace_in_file: missing required parameter 'path'"}
	}
	if oldText == "" {
		return ToolResult{Error: "replace_in_file: missing required parameter 'old_text'"}
	}

	absP, err := th.safePath(path)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("replace_in_file: %v", err)}
	}
	data, err := os.ReadFile(absP)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("replace_in_file: %v", err)}
	}

	content := string(data)

	// Strip leading newline from old_text/new_text (XML artifact)
	oldText = strings.TrimPrefix(oldText, "\n")
	newText = strings.TrimPrefix(newText, "\n")

	// Try exact match first
	if strings.Contains(content, oldText) {
		count := strings.Count(content, oldText)
		if count > 1 {
			logWarn("replace_in_file: %d matches found, replacing first only", count)
		}
		content = strings.Replace(content, oldText, newText, 1)
		if err := os.WriteFile(absP, []byte(content), 0644); err != nil {
			return ToolResult{Error: fmt.Sprintf("replace_in_file: write failed: %v", err)}
		}
		return ToolResult{Output: fmt.Sprintf("Replaced in %s", path)}
	}

	// Strategy 2: Normalize trailing whitespace
	if result := th.tryNormalizedReplace(content, oldText, newText, absP); result != nil {
		return *result
	}

	// Strategy 3: Indentation agnostic
	if result := th.tryIndentAgnosticReplace(content, oldText, newText, absP); result != nil {
		return *result
	}

	return ToolResult{Error: fmt.Sprintf("replace_in_file: old_text not found in %s. Ensure the text matches exactly including whitespace.", path)}
}

func (th *ToolHandlers) tryNormalizedReplace(content, oldText, newText, absP string) *ToolResult {
	// Strip trailing whitespace from each line for comparison
	normalizeLines := func(s string) string {
		lines := strings.Split(s, "\n")
		for i, l := range lines {
			lines[i] = strings.TrimRight(l, " \t\r")
		}
		return strings.Join(lines, "\n")
	}
	normContent := normalizeLines(content)
	normOld := normalizeLines(oldText)

	idx := strings.Index(normContent, normOld)
	if idx < 0 {
		return nil
	}

	// Only proceed if the match starts at a line boundary
	if idx > 0 && normContent[idx-1] != '\n' {
		return nil
	}

	// Map the normalized match position back to original content by line number.
	// Count which line the match starts on in normalized content.
	lineNum := strings.Count(normContent[:idx], "\n")
	oldLineCount := strings.Count(normOld, "\n") + 1

	contentLines := strings.Split(content, "\n")
	if lineNum+oldLineCount > len(contentLines) {
		return nil
	}

	// Replace matched lines with newText
	newLines := strings.Split(newText, "\n")
	result := make([]string, 0, len(contentLines)-oldLineCount+len(newLines))
	result = append(result, contentLines[:lineNum]...)
	result = append(result, newLines...)
	result = append(result, contentLines[lineNum+oldLineCount:]...)

	final := strings.Join(result, "\n")
	if err := os.WriteFile(absP, []byte(final), 0644); err != nil {
		return &ToolResult{Error: fmt.Sprintf("replace_in_file: write failed: %v", err)}
	}
	r := ToolResult{Output: fmt.Sprintf("Replaced in %s (trailing whitespace normalized)", filepath.Base(absP))}
	return &r
}

func (th *ToolHandlers) tryIndentAgnosticReplace(content, oldText, newText, absP string) *ToolResult {
	// Strip all leading whitespace from each line
	stripIndent := func(s string) string {
		lines := strings.Split(s, "\n")
		for i, l := range lines {
			lines[i] = strings.TrimLeft(l, " \t")
		}
		return strings.Join(lines, "\n")
	}

	strippedContent := stripIndent(content)
	strippedOld := stripIndent(oldText)

	idx := strings.Index(strippedContent, strippedOld)
	if idx < 0 {
		return nil
	}

	// Find the corresponding position in the original content
	// by counting newlines up to the match position
	lineNum := strings.Count(strippedContent[:idx], "\n")
	contentLines := strings.Split(content, "\n")
	oldLines := strings.Split(oldText, "\n")

	if lineNum+len(oldLines) > len(contentLines) {
		return nil
	}

	// Detect the indentation of the first matched line
	indent := ""
	if lineNum < len(contentLines) {
		trimmed := strings.TrimLeft(contentLines[lineNum], " \t")
		indent = contentLines[lineNum][:len(contentLines[lineNum])-len(trimmed)]
	}

	// Detect the base indentation of old_text (minimum non-empty line indent)
	oldIndent := ""
	for _, l := range oldLines {
		if strings.TrimSpace(l) == "" {
			continue
		}
		trimmed := strings.TrimLeft(l, " \t")
		li := l[:len(l)-len(trimmed)]
		if oldIndent == "" || len(li) < len(oldIndent) {
			oldIndent = li
		}
	}

	// Apply indentation to new_text preserving relative structure
	newLines := strings.Split(newText, "\n")
	// Detect base indentation of new_text
	newBaseIndent := ""
	for _, l := range newLines {
		if strings.TrimSpace(l) == "" {
			continue
		}
		trimmed := strings.TrimLeft(l, " \t")
		li := l[:len(l)-len(trimmed)]
		if newBaseIndent == "" || len(li) < len(newBaseIndent) {
			newBaseIndent = li
		}
	}
	for i, l := range newLines {
		if strings.TrimSpace(l) != "" {
			// Strip the new_text base indent, add the target base indent + relative remainder
			relative := strings.TrimPrefix(l, newBaseIndent)
			newLines[i] = indent + relative
		}
	}

	// Replace lines
	result := make([]string, 0, len(contentLines))
	result = append(result, contentLines[:lineNum]...)
	result = append(result, newLines...)
	result = append(result, contentLines[lineNum+len(oldLines):]...)

	final := strings.Join(result, "\n")
	if err := os.WriteFile(absP, []byte(final), 0644); err != nil {
		return &ToolResult{Error: fmt.Sprintf("replace_in_file: write failed: %v", err)}
	}
	r := ToolResult{Output: fmt.Sprintf("Replaced in %s (indentation adjusted)", filepath.Base(absP))}
	return &r
}

// ── replace_between_anchors ─────────────────────────────────────────────

func (th *ToolHandlers) replaceBetweenAnchors(params map[string]string) ToolResult {
	path := params["path"]
	startAnchor := params["start_anchor"]
	endAnchor := params["end_anchor"]
	replacement := params["replacement"]

	if path == "" || startAnchor == "" || endAnchor == "" {
		return ToolResult{Error: "replace_between_anchors: missing required parameter"}
	}

	absP, err := th.safePath(path)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("replace_between_anchors: %v", err)}
	}
	data, err := os.ReadFile(absP)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("replace_between_anchors: %v", err)}
	}

	content := string(data)
	startIdx := strings.Index(content, startAnchor)
	if startIdx < 0 {
		return ToolResult{Error: fmt.Sprintf("replace_between_anchors: start_anchor not found in %s", path)}
	}

	afterStart := startIdx + len(startAnchor)
	endIdx := strings.Index(content[afterStart:], endAnchor)
	if endIdx < 0 {
		return ToolResult{Error: fmt.Sprintf("replace_between_anchors: end_anchor not found in %s", path)}
	}
	endIdx += afterStart

	newContent := content[:afterStart] + "\n" + strings.TrimPrefix(replacement, "\n") + "\n" + content[endIdx:]
	if err := os.WriteFile(absP, []byte(newContent), 0644); err != nil {
		return ToolResult{Error: fmt.Sprintf("replace_between_anchors: write failed: %v", err)}
	}

	return ToolResult{Output: fmt.Sprintf("Replaced between anchors in %s", path)}
}

// ── execute_command ─────────────────────────────────────────────────────

func (th *ToolHandlers) executeCommand(params map[string]string) ToolResult {
	command := params["command"]
	if command == "" {
		return ToolResult{Error: "execute_command: missing required parameter 'command'"}
	}

	background := strings.TrimSpace(strings.ToLower(params["background"])) == "true"

	if background {
		return th.executeBackground(command)
	}

	// Foreground execution with timeout
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("powershell", "-NoProfile", "-Command", command)
	} else {
		cmd = exec.Command("bash", "-c", command)
	}
	cmd.Dir = th.Workspace

	// Process group isolation
	setProcGroup(cmd)

	var outBuf strings.Builder
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return ToolResult{Error: fmt.Sprintf("execute_command: start failed: %v", err)}
	}

	go func() { done <- cmd.Wait() }()

	timeout := 120 * time.Second
	select {
	case err := <-done:
		output := outBuf.String()
		output = stripANSI(output)
		if len(output) > 100000 {
			output = output[:100000] + "\n... (output truncated)"
		}
		if err != nil {
			exitCode := -1
			if cmd.ProcessState != nil {
				exitCode = cmd.ProcessState.ExitCode()
			}
			return ToolResult{
				Output: output,
				Error:  fmt.Sprintf("Command exited with code %d", exitCode),
			}
		}
		return ToolResult{Output: output}
	case <-time.After(timeout):
		cmd.Process.Kill()
		// Wait for the pipe-copy goroutine to finish before reading the buffer
		<-done
		return ToolResult{
			Output: outBuf.String(),
			Error:  fmt.Sprintf("Command timed out after %v", timeout),
		}
	}
}

func (th *ToolHandlers) executeBackground(command string) ToolResult {
	th.mu.Lock()
	id := th.nextBgID
	th.nextBgID++
	th.mu.Unlock()

	logPath := filepath.Join(th.spillDir, fmt.Sprintf("bg_%d.log", id))
	logFile, err := os.Create(logPath)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("execute_command: cannot create log: %v", err)}
	}

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("powershell", "-NoProfile", "-Command", command)
	} else {
		cmd = exec.Command("bash", "-c", command)
	}
	cmd.Dir = th.Workspace
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	setProcGroup(cmd)

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return ToolResult{Error: fmt.Sprintf("execute_command: start failed: %v", err)}
	}

	proc := &bgProcess{
		ID:        id,
		PID:       cmd.Process.Pid,
		Command:   command,
		StartTime: time.Now(),
		LogPath:   logPath,
	}

	th.mu.Lock()
	th.bgProcs[id] = proc
	th.mu.Unlock()

	// Monitor in background
	go func() {
		cmd.Wait()
		logFile.Close()
		th.mu.Lock()
		proc.Done = true
		if cmd.ProcessState != nil {
			proc.ExitCode = cmd.ProcessState.ExitCode()
		}
		th.mu.Unlock()
	}()

	return ToolResult{Output: fmt.Sprintf("Background process started: ID=%d PID=%d\nCommand: %s\nLog: %s", id, cmd.Process.Pid, command, logPath)}
}

// ── list_files ──────────────────────────────────────────────────────────

func (th *ToolHandlers) listFiles(params map[string]string) ToolResult {
	path := params["path"]
	if path == "" {
		path = "."
	}
	recursive := strings.TrimSpace(strings.ToLower(params["recursive"])) == "true"

	absP, err := th.safePath(path)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("list_files: %v", err)}
	}

	var entries []string
	if recursive {
		filepath.Walk(absP, func(p string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(absP, p)
			if rel == "." {
				return nil
			}
			if info.IsDir() {
				if skipDirs[info.Name()] {
					return filepath.SkipDir
				}
				entries = append(entries, rel+"/")
			} else {
				entries = append(entries, rel)
			}
			if len(entries) > 2000 {
				return fmt.Errorf("truncated")
			}
			return nil
		})
	} else {
		dirEntries, err := os.ReadDir(absP)
		if err != nil {
			return ToolResult{Error: fmt.Sprintf("list_files: %v", err)}
		}
		for _, e := range dirEntries {
			if e.IsDir() {
				entries = append(entries, e.Name()+"/")
			} else {
				entries = append(entries, e.Name())
			}
		}
	}

	sort.Strings(entries)
	return ToolResult{Output: strings.Join(entries, "\n")}
}

// ── search_files ────────────────────────────────────────────────────────

func (th *ToolHandlers) searchFiles(params map[string]string) ToolResult {
	searchPath := params["path"]
	pattern := params["regex"]
	filePattern := params["file_pattern"]
	if searchPath == "" || pattern == "" {
		return ToolResult{Error: "search_files: missing required parameter 'path' or 'regex'"}
	}

	re, err := regexp.Compile(pattern)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("search_files: invalid regex: %v", err)}
	}

	absP, err := th.safePath(searchPath)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("search_files: %v", err)}
	}
	var results []string
	maxResults := 100

	filepath.Walk(absP, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			if info != nil && info.IsDir() && skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if len(results) >= maxResults {
			return fmt.Errorf("truncated")
		}

		// Check file pattern
		if filePattern != "" {
			matched, _ := filepath.Match(filePattern, info.Name())
			if !matched {
				return nil
			}
		}

		// Skip binary files
		ext := strings.ToLower(filepath.Ext(path))
		if binaryExts[ext] {
			return nil
		}

		// Skip large files
		if info.Size() > 2_000_000 {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		lines := strings.Split(string(data), "\n")
		rel, _ := filepath.Rel(th.Workspace, path)
		for i, line := range lines {
			if re.MatchString(line) {
				// Show context: 1 line before and after
				start := i - 1
				if start < 0 {
					start = 0
				}
				end := i + 2
				if end > len(lines) {
					end = len(lines)
				}
				results = append(results, fmt.Sprintf("%s:%d:", rel, i+1))
				for j := start; j < end; j++ {
					marker := " "
					if j == i {
						marker = ">"
					}
					results = append(results, fmt.Sprintf("  %s %4d | %s", marker, j+1, lines[j]))
				}
				results = append(results, "")
			}
		}
		return nil
	})

	if len(results) == 0 {
		return ToolResult{Output: fmt.Sprintf("No matches found for /%s/ in %s", pattern, searchPath)}
	}
	return ToolResult{Output: strings.Join(results, "\n")}
}

// ── background process management ───────────────────────────────────────

func (th *ToolHandlers) listBgProcs() ToolResult {
	th.mu.Lock()
	defer th.mu.Unlock()

	if len(th.bgProcs) == 0 {
		return ToolResult{Output: "No background processes."}
	}

	var sb strings.Builder
	for _, p := range th.bgProcs {
		status := "running"
		if p.Done {
			status = fmt.Sprintf("exited (%d)", p.ExitCode)
		}
		elapsed := time.Since(p.StartTime).Round(time.Second)
		sb.WriteString(fmt.Sprintf("ID=%d  PID=%d  %s  %s  %s\n", p.ID, p.PID, status, elapsed, p.Command))
	}
	return ToolResult{Output: sb.String()}
}

func (th *ToolHandlers) checkBgProc(params map[string]string) ToolResult {
	idStr := params["id"]
	id, err := strconv.Atoi(strings.TrimSpace(idStr))
	if err != nil {
		return ToolResult{Error: "check_background_process: invalid ID"}
	}

	th.mu.Lock()
	proc, ok := th.bgProcs[id]
	th.mu.Unlock()

	if !ok {
		return ToolResult{Error: fmt.Sprintf("check_background_process: no process with ID %d", id)}
	}

	nLines := 50
	if n, ok := params["lines"]; ok {
		if v, err := strconv.Atoi(strings.TrimSpace(n)); err == nil {
			nLines = v
		}
	}

	data, err := os.ReadFile(proc.LogPath)
	if err != nil {
		return ToolResult{Output: fmt.Sprintf("Process ID=%d PID=%d — no output yet", id, proc.PID)}
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) > nLines {
		lines = lines[len(lines)-nLines:]
	}

	status := "running"
	if proc.Done {
		status = fmt.Sprintf("exited (code %d)", proc.ExitCode)
	}

	return ToolResult{Output: fmt.Sprintf("Process ID=%d PID=%d Status=%s\n\n%s", id, proc.PID, status, strings.Join(lines, "\n"))}
}

func (th *ToolHandlers) stopBgProc(params map[string]string) ToolResult {
	idStr := params["id"]
	id, err := strconv.Atoi(strings.TrimSpace(idStr))
	if err != nil {
		return ToolResult{Error: "stop_background_process: invalid ID"}
	}

	th.mu.Lock()
	proc, ok := th.bgProcs[id]
	th.mu.Unlock()

	if !ok {
		return ToolResult{Error: fmt.Sprintf("stop_background_process: no process with ID %d", id)}
	}

	if proc.Done {
		return ToolResult{Output: fmt.Sprintf("Process ID=%d already exited", id)}
	}

	if err := killProcessTree(proc.PID); err != nil {
		return ToolResult{Error: fmt.Sprintf("stop_background_process: kill failed: %v", err)}
	}

	return ToolResult{Output: fmt.Sprintf("Process ID=%d (PID %d) terminated", id, proc.PID)}
}

// ── retrieve_tool_result ────────────────────────────────────────────────

func (th *ToolHandlers) retrieveToolResult(params map[string]string) ToolResult {
	resultID := params["result_id"]
	if resultID == "" {
		return ToolResult{Error: "retrieve_tool_result: missing result_id"}
	}

	// Look for spilled files matching the ID
	entries, _ := os.ReadDir(th.spillDir)
	for _, e := range entries {
		if strings.Contains(e.Name(), resultID) {
			data, err := os.ReadFile(filepath.Join(th.spillDir, e.Name()))
			if err != nil {
				return ToolResult{Error: fmt.Sprintf("retrieve_tool_result: %v", err)}
			}
			return ToolResult{Output: string(data)}
		}
	}

	return ToolResult{Error: fmt.Sprintf("retrieve_tool_result: result '%s' not found", resultID)}
}

// ── attempt_completion ──────────────────────────────────────────────────

func (th *ToolHandlers) attemptCompletion(params map[string]string) ToolResult {
	result := params["result"]
	command := params["command"]

	var sb strings.Builder
	if result != "" {
		sb.WriteString(result)
	}
	if command != "" {
		sb.WriteString("\n\nSuggested command: " + command)
	}
	return ToolResult{Output: sb.String()}
}

// ── helpers ─────────────────────────────────────────────────────────────

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

func stripANSI(s string) string {
	return ansiRe.ReplaceAllString(s, "")
}
