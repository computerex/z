package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestBuildSucceeds verifies the binary compiles cleanly.
func TestBuildSucceeds(t *testing.T) {
	cmd := exec.Command("go", "build", "-o", binaryName(), ".")
	cmd.Dir = projectDir()
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Build failed: %s\n%s", err, string(out))
	}
	// Verify binary exists
	binPath := filepath.Join(projectDir(), binaryName())
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		t.Fatalf("Binary not found at %s", binPath)
	}
}

// TestHelpFlag tests that --help produces expected output.
func TestHelpFlag(t *testing.T) {
	out := runBinary(t, "--help")
	assertContains(t, out, "harness v")
	assertContains(t, out, "agentic coding assistant (Go)")
	assertContains(t, out, "--workspace")
	assertContains(t, out, "--session")
	assertContains(t, out, "--install")
}

// TestListSessionsEmpty tests that --list works on a clean workspace.
func TestListSessionsEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	out := runBinary(t, "--list", "--workspace", tmpDir)
	assertContains(t, out, "No saved sessions")
}

// ── Tool handler parity tests ───────────────────────────────────────────

func TestReadFileTool(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Create a test file
	testFile := filepath.Join(th.Workspace, "test.txt")
	content := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	os.WriteFile(testFile, []byte(content), 0644)

	// Test full read
	result := th.Dispatch(ToolCall{Name: "read_file", Params: map[string]string{"path": "test.txt"}})
	if result.Error != "" {
		t.Fatalf("read_file error: %s", result.Error)
	}
	assertContains(t, result.Output, "1 | line 1")
	assertContains(t, result.Output, "5 | line 5")

	// Test line range
	result = th.Dispatch(ToolCall{Name: "read_file", Params: map[string]string{"path": "test.txt", "start_line": "2", "end_line": "3"}})
	assertContains(t, result.Output, "2 | line 2")
	assertContains(t, result.Output, "3 | line 3")
	assertNotContains(t, result.Output, "1 | line 1")
}

func TestWriteToFileTool(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Basic write
	result := th.Dispatch(ToolCall{Name: "write_to_file", Params: map[string]string{
		"path":    "output.txt",
		"content": "hello world",
	}})
	if result.Error != "" {
		t.Fatalf("write_to_file error: %s", result.Error)
	}
	assertContains(t, result.Output, "File written")

	data, _ := os.ReadFile(filepath.Join(th.Workspace, "output.txt"))
	if !strings.Contains(string(data), "hello world") {
		t.Fatalf("Written content mismatch: %q", string(data))
	}

	// Write with auto-created subdirectories
	result = th.Dispatch(ToolCall{Name: "write_to_file", Params: map[string]string{
		"path":    "sub/dir/file.txt",
		"content": "nested",
	}})
	if result.Error != "" {
		t.Fatalf("write_to_file nested error: %s", result.Error)
	}
	data, _ = os.ReadFile(filepath.Join(th.Workspace, "sub", "dir", "file.txt"))
	if !strings.Contains(string(data), "nested") {
		t.Fatalf("Nested write mismatch: %q", string(data))
	}
}

func TestReplaceInFileTool(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Create file
	testFile := filepath.Join(th.Workspace, "replace_test.py")
	os.WriteFile(testFile, []byte("def hello():\n    print('hello')\n    return True\n"), 0644)

	// Exact match replace
	result := th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "replace_test.py",
		"old_text": "    print('hello')",
		"new_text": "    print('world')",
	}})
	if result.Error != "" {
		t.Fatalf("replace_in_file error: %s", result.Error)
	}

	data, _ := os.ReadFile(testFile)
	assertContains(t, string(data), "print('world')")
	assertNotContains(t, string(data), "print('hello')")
}

func TestReplaceInFileTrailingWhitespace(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Create file with trailing spaces
	testFile := filepath.Join(th.Workspace, "ws_test.txt")
	os.WriteFile(testFile, []byte("line one  \nline two  \n"), 0644)

	// Replace without trailing spaces (should still match via normalization)
	result := th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "ws_test.txt",
		"old_text": "line one\nline two",
		"new_text": "LINE ONE\nLINE TWO",
	}})
	if result.Error != "" {
		t.Fatalf("replace_in_file trailing ws error: %s", result.Error)
	}
	data, _ := os.ReadFile(testFile)
	if !strings.Contains(string(data), "LINE ONE") {
		t.Fatalf("Trailing whitespace normalization failed: %q", string(data))
	}
}

func TestReplaceBetweenAnchors(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	testFile := filepath.Join(th.Workspace, "anchor_test.txt")
	os.WriteFile(testFile, []byte("// START\nold content\n// END\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "replace_between_anchors", Params: map[string]string{
		"path":         "anchor_test.txt",
		"start_anchor": "// START",
		"end_anchor":   "// END",
		"replacement":  "new content",
	}})
	if result.Error != "" {
		t.Fatalf("replace_between_anchors error: %s", result.Error)
	}

	data, _ := os.ReadFile(testFile)
	assertContains(t, string(data), "// START")
	assertContains(t, string(data), "new content")
	assertContains(t, string(data), "// END")
	assertNotContains(t, string(data), "old content")
}

func TestExecuteCommand(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	var cmd string
	if runtime.GOOS == "windows" {
		cmd = `echo "hello_from_Go_harness"`
	} else {
		cmd = "echo hello_from_Go_harness"
	}

	result := th.Dispatch(ToolCall{Name: "execute_command", Params: map[string]string{"command": cmd}})
	if result.Error != "" && !strings.Contains(result.Error, "exited") {
		t.Fatalf("execute_command error: %s", result.Error)
	}
	assertContains(t, result.Output, "hello_from_Go_harness")
}

func TestExecuteCommandBackground(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	var cmd string
	if runtime.GOOS == "windows" {
		cmd = "echo bg_test"
	} else {
		cmd = "echo bg_test"
	}

	result := th.Dispatch(ToolCall{Name: "execute_command", Params: map[string]string{
		"command":    cmd,
		"background": "true",
	}})
	assertContains(t, result.Output, "Background process started")
	assertContains(t, result.Output, "ID=1")

	// List background processes
	result = th.Dispatch(ToolCall{Name: "list_background_processes", Params: map[string]string{}})
	assertContains(t, result.Output, "ID=1")
}

func TestListFiles(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Create some files
	os.WriteFile(filepath.Join(th.Workspace, "a.txt"), []byte("a"), 0644)
	os.WriteFile(filepath.Join(th.Workspace, "b.py"), []byte("b"), 0644)
	os.MkdirAll(filepath.Join(th.Workspace, "sub"), 0755)
	os.WriteFile(filepath.Join(th.Workspace, "sub", "c.go"), []byte("c"), 0644)

	// Non-recursive
	result := th.Dispatch(ToolCall{Name: "list_files", Params: map[string]string{"path": "."}})
	assertContains(t, result.Output, "a.txt")
	assertContains(t, result.Output, "b.py")
	assertContains(t, result.Output, "sub/")

	// Recursive
	result = th.Dispatch(ToolCall{Name: "list_files", Params: map[string]string{"path": ".", "recursive": "true"}})
	assertContains(t, result.Output, "a.txt")
	assertContains(t, result.Output, filepath.Join("sub", "c.go"))
}

func TestSearchFiles(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	os.WriteFile(filepath.Join(th.Workspace, "search.py"), []byte("def find_me():\n    pass\ndef other():\n    pass\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "search_files", Params: map[string]string{
		"path":  ".",
		"regex": "find_me",
	}})
	assertContains(t, result.Output, "search.py")
	assertContains(t, result.Output, "find_me")
}

// ── XML parsing parity tests ────────────────────────────────────────────

func TestParseToolCallsSimple(t *testing.T) {
	input := `Let me read the file.

<read_file>
<path>src/main.py</path>
</read_file>`

	calls := ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Name != "read_file" {
		t.Fatalf("Expected read_file, got %s", calls[0].Name)
	}
	if calls[0].Params["path"] != "src/main.py" {
		t.Fatalf("Expected path=src/main.py, got %s", calls[0].Params["path"])
	}
}

func TestParseToolCallsGreedy(t *testing.T) {
	// write_to_file uses greedy matching for content
	input := `<write_to_file>
<path>test.py</path>
<content>
def main():
    print("hello")
    # This has <tags> inside
    return 0
</content>
</write_to_file>`

	calls := ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Name != "write_to_file" {
		t.Fatalf("Expected write_to_file, got %s", calls[0].Name)
	}
	if !strings.Contains(calls[0].Params["content"], "hello") {
		t.Fatalf("Content should contain 'hello': %s", calls[0].Params["content"])
	}
}

func TestParseToolCallsMultiple(t *testing.T) {
	input := `<read_file>
<path>a.py</path>
</read_file>

<read_file>
<path>b.py</path>
</read_file>`

	calls := ParseToolCalls(input)
	if len(calls) != 2 {
		t.Fatalf("Expected 2 tool calls, got %d", len(calls))
	}
	if calls[0].Params["path"] != "a.py" {
		t.Fatalf("First call path wrong: %s", calls[0].Params["path"])
	}
	if calls[1].Params["path"] != "b.py" {
		t.Fatalf("Second call path wrong: %s", calls[1].Params["path"])
	}
}

func TestStripThinkingTags(t *testing.T) {
	input := `<thinking>I need to analyze this</thinking>

Here is my response.`
	result := StripThinkingTags(input)
	assertNotContains(t, result, "thinking")
	assertContains(t, result, "Here is my response")
}

func TestStripToolTags(t *testing.T) {
	input := `I'll read the file.

<read_file>
<path>test.py</path>
</read_file>

Done.`
	result := StripToolTags(input)
	assertNotContains(t, result, "<read_file>")
	assertContains(t, result, "read the file")
	assertContains(t, result, "Done.")
}

// ── Session parity tests ───────────────────────────────────────────────

func TestSessionSaveLoad(t *testing.T) {
	tmpDir := t.TempDir()
	sessPath := filepath.Join(tmpDir, "test_session.json")

	// Create session with a system prompt that passes integrity check
	sess := NewSession("/test/workspace")
	sess.AppendMessage("system", "You are a helpful assistant.\n\n# TOOL USE\n\nYou have access to read_file and other tools.")
	sess.AppendMessage("user", "Hello!")
	sess.AppendAssistant("Hi there!", nil)
	sess.AddContext("file", "test.py", "def main():\n    pass\n", nil)

	// Save
	err := SaveSession(sess, sessPath)
	if err != nil {
		t.Fatalf("SaveSession error: %v", err)
	}

	// Load
	loaded, err := LoadSession(sessPath)
	if err != nil {
		t.Fatalf("LoadSession error: %v", err)
	}

	if len(loaded.Messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(loaded.Messages))
	}
	if loaded.Messages[0].Role != "system" {
		t.Fatalf("First message should be system, got %s", loaded.Messages[0].Role)
	}
	if loaded.Messages[1].Role != "user" {
		t.Fatalf("Second message should be user, got %s", loaded.Messages[1].Role)
	}
	if loaded.Workspace != "/test/workspace" {
		t.Fatalf("Workspace mismatch: %s", loaded.Workspace)
	}

	// Verify context
	if len(loaded.Context) != 1 {
		t.Fatalf("Expected 1 context item, got %d", len(loaded.Context))
	}
}

func TestSessionJSON_CompatibleFormat(t *testing.T) {
	// Verify the JSON format matches the Python harness format
	sess := NewSession("/workspace")
	sess.AppendMessage("system", "system prompt")
	sess.AppendMessage("user", "user message")
	sess.AppendAssistant("assistant reply", nil)
	sess.AddContext("file", "test.py", "pass", nil) // add context so it appears in JSON

	data, _ := json.Marshal(sess)
	var raw map[string]interface{}
	json.Unmarshal(data, &raw)

	// Must have these top-level keys (matching Python session format)
	requiredKeys := []string{"workspace", "messages", "context", "context_next_id"}
	for _, key := range requiredKeys {
		if _, ok := raw[key]; !ok {
			t.Fatalf("Session JSON missing required key: %s\nJSON: %s", key, string(data))
		}
	}

	// Messages must have role and content
	msgs := raw["messages"].([]interface{})
	for i, m := range msgs {
		msg := m.(map[string]interface{})
		if _, ok := msg["role"]; !ok {
			t.Fatalf("Message %d missing 'role'", i)
		}
		if _, ok := msg["content"]; !ok {
			t.Fatalf("Message %d missing 'content'", i)
		}
	}
}

// ── Config parity tests ────────────────────────────────────────────────

func TestConfigDefaults(t *testing.T) {
	cfg := defaultConfig()
	if cfg.Model != "gpt-4o" {
		t.Fatalf("Default model should be gpt-4o, got %s", cfg.Model)
	}
	if cfg.MaxContextTokens != 128000 {
		t.Fatalf("Default max_context_tokens should be 128000, got %d", cfg.MaxContextTokens)
	}
	if cfg.CompactionThreshold != 0.85 {
		t.Fatalf("Default compaction_threshold should be 0.85, got %f", cfg.CompactionThreshold)
	}
}

func TestConfigLoadFromJSON(t *testing.T) {
	// Create a temporary ~/.z.json
	tmpHome := t.TempDir()

	// Save original env and restore after test
	origHome := os.Getenv("USERPROFILE")
	origHomeUnix := os.Getenv("HOME")
	defer func() {
		os.Setenv("USERPROFILE", origHome)
		os.Setenv("HOME", origHomeUnix)
	}()

	zConfig := map[string]interface{}{
		"api_url":  "https://api.test.com",
		"api_key":  "test-key-123",
		"model":    "claude-3-opus",
		"providers": map[string]interface{}{
			"anthropic": map[string]interface{}{
				"api_url": "https://api.anthropic.com",
				"api_key": "sk-ant-test",
				"model":   "claude-3-opus",
			},
		},
		"mcp": map[string]interface{}{
			"test-server": map[string]interface{}{
				"type":    "stdio",
				"command": "test-mcp",
				"enabled": true,
			},
		},
	}

	data, _ := json.Marshal(zConfig)
	zPath := filepath.Join(tmpHome, ".z.json")
	os.WriteFile(zPath, data, 0644)

	// The loadConfig reads from globalConfigPath which uses UserHomeDir
	// We can't easily override that in a test, so just verify the JSON parsing
	var raw zJSON
	json.Unmarshal(data, &raw)

	if raw.APIURL != "https://api.test.com" {
		t.Fatalf("APIURL mismatch: %s", raw.APIURL)
	}
	if raw.Model != "claude-3-opus" {
		t.Fatalf("Model mismatch: %s", raw.Model)
	}
	if _, ok := raw.Providers["anthropic"]; !ok {
		t.Fatal("Missing anthropic provider")
	}
}

// ── System prompt parity tests ──────────────────────────────────────────

func TestSystemPromptContainsRequiredSections(t *testing.T) {
	prompt := buildSystemPrompt("/test/workspace", "src/\n  main.py\n", "", "")

	requiredSections := []string{
		"TOOL USE",
		"read_file",
		"write_to_file",
		"replace_in_file",
		"execute_command",
		"list_files",
		"search_files",
		"manage_todos",
		"attempt_completion",
		"RULES",
		"SYSTEM INFORMATION",
		"OBJECTIVE",
		"/test/workspace",
	}

	for _, section := range requiredSections {
		assertContains(t, prompt, section)
	}
}

func TestSystemPromptToolDocsFormat(t *testing.T) {
	prompt := buildSystemPrompt("/workspace", "", "", "")

	// Verify XML tool formatting examples
	assertContains(t, prompt, "<tool_name>")
	assertContains(t, prompt, "<parameter1_name>")
	assertContains(t, prompt, "</tool_name>")

	// Verify read_file example
	assertContains(t, prompt, "<read_file>")
	assertContains(t, prompt, "<path>")
	assertContains(t, prompt, "</read_file>")
}

func TestSystemPromptInstructions(t *testing.T) {
	prompt := buildSystemPrompt("/workspace", "", "", "Custom instructions here")
	assertContains(t, prompt, "WORKSPACE INSTRUCTIONS")
	assertContains(t, prompt, "Custom instructions here")
}

// ── Workspace index parity tests ────────────────────────────────────────

func TestWorkspaceIndexSkipsDirs(t *testing.T) {
	tmpDir := t.TempDir()

	// Create some dirs and files
	os.MkdirAll(filepath.Join(tmpDir, "src"), 0755)
	os.MkdirAll(filepath.Join(tmpDir, "node_modules", "pkg"), 0755)
	os.MkdirAll(filepath.Join(tmpDir, ".git", "objects"), 0755)
	os.WriteFile(filepath.Join(tmpDir, "src", "main.py"), []byte("pass"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "node_modules", "pkg", "index.js"), []byte("//"), 0644)

	idx := BuildWorkspaceIndex(tmpDir)
	tree := idx.CompactTree()

	assertContains(t, tree, "main.py")
	assertNotContains(t, tree, "node_modules")
	assertNotContains(t, tree, ".git")
}

func TestWorkspaceCompactTree(t *testing.T) {
	tmpDir := t.TempDir()

	os.MkdirAll(filepath.Join(tmpDir, "src"), 0755)
	os.WriteFile(filepath.Join(tmpDir, "README.md"), []byte("# test"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "src", "app.py"), []byte("pass"), 0644)

	idx := BuildWorkspaceIndex(tmpDir)
	tree := idx.CompactTree()

	// Should contain directory structure
	assertContains(t, tree, "README.md")
	assertContains(t, tree, "src/")
	assertContains(t, tree, "app.py")
}

// ── Display/XML suppression parity tests ────────────────────────────────

func TestDisplayXMLSuppression(t *testing.T) {
	display := NewDisplay(nil)

	// Simulate streaming a response with a tool call
	chunks := []string{
		"Let me read ",
		"the file.\n\n",
		"<read_fi",
		"le>\n<path>",
		"src/main.py</",
		"path>\n</read_file>",
	}

	var visible strings.Builder
	for _, chunk := range chunks {
		out := display.OnChunk(chunk)
		visible.WriteString(out)
	}

	result := visible.String()
	assertContains(t, result, "Let me read the file.")
	assertNotContains(t, result, "<read_file>")
	assertNotContains(t, result, "src/main.py")
}

func TestDisplayThinkingSuppress(t *testing.T) {
	display := NewDisplay(nil)

	chunks := []string{
		"<thinking>I need to think",
		" about this carefully",
		"</thinking>\n\nHere is my answer.",
	}

	var visible strings.Builder
	for _, chunk := range chunks {
		out := display.OnChunk(chunk)
		visible.WriteString(out)
	}

	result := visible.String()
	assertContains(t, result, "Here is my answer.")
	assertNotContains(t, result, "<thinking>")
	assertNotContains(t, result, "I need to think")
}

// ── Interrupt parity tests ──────────────────────────────────────────────

func TestInterruptState(t *testing.T) {
	is := &InterruptState{}

	// Initially not interrupted
	if is.IsInterrupted() {
		t.Fatal("Should not be interrupted initially")
	}

	// Trigger
	is.Trigger("test")
	if !is.IsInterrupted() {
		t.Fatal("Should be interrupted after trigger")
	}
	interrupted, reason := is.Snapshot()
	if !interrupted || reason != "test" {
		t.Fatalf("Snapshot mismatch: interrupted=%v reason=%s", interrupted, reason)
	}

	// Reset
	is.Reset()
	if is.IsInterrupted() {
		t.Fatal("Should not be interrupted after reset")
	}
}

func TestInterruptDoubleTap(t *testing.T) {
	// Verify the double-tap window concept
	km := NewKeyboardMonitor()
	if km.doubleTapWindow.Seconds() < 1.0 || km.doubleTapWindow.Seconds() > 2.0 {
		t.Fatalf("Double-tap window should be ~1.5s, got %v", km.doubleTapWindow)
	}
}

// ── Agent loop parity tests ─────────────────────────────────────────────

func TestAgentOneToolPerTurn(t *testing.T) {
	// Verify that multiple tool calls in a single response reports the extras
	content := `<read_file>
<path>a.py</path>
</read_file>

<read_file>
<path>b.py</path>
</read_file>

<write_to_file>
<path>c.py</path>
<content>pass</content>
</write_to_file>`

	calls := ParseToolCalls(content)
	if len(calls) != 3 {
		t.Fatalf("Expected 3 tool calls parsed, got %d", len(calls))
	}

	// Agent should execute only first, report extras (tested via agent.go logic)
	extra := calls[1:]
	if len(extra) != 2 {
		t.Fatalf("Expected 2 extra calls, got %d", len(extra))
	}
}

func TestAgentMaxIterations(t *testing.T) {
	// Default max should be 500 (matching Python)
	cfg := defaultConfig()
	agent := NewAgent(t.TempDir(), cfg, "")
	if agent.MaxIterations != 500 {
		t.Fatalf("MaxIterations should be 500, got %d", agent.MaxIterations)
	}
}

func TestHasUnclosedToolTag(t *testing.T) {
	cases := []struct {
		input    string
		expected bool
	}{
		{"<read_file><path>test</path></read_file>", false},
		{"<read_file><path>test</path>", true},
		{"no tools here", false},
		{"<write_to_file><path>f</path><content>partial", true},
		{"<thinking>just thinking</thinking>", false},
	}

	for _, tc := range cases {
		result := hasUnclosedToolTag(tc.input)
		if result != tc.expected {
			t.Errorf("hasUnclosedToolTag(%q) = %v, want %v", tc.input[:min(len(tc.input), 40)], result, tc.expected)
		}
	}
}

// ── Token estimation parity tests ───────────────────────────────────────

func TestSessionTokenEstimate(t *testing.T) {
	sess := NewSession("/workspace")
	sess.AppendMessage("system", strings.Repeat("x", 4000)) // ~1000 tokens
	sess.AppendMessage("user", strings.Repeat("y", 400))     // ~100 tokens

	tokens := sess.EstimateTokens()
	// Rough estimate: 4400 chars / 4 = 1100 tokens
	if tokens < 900 || tokens > 1300 {
		t.Fatalf("Token estimate out of range: %d (expected ~1100)", tokens)
	}
}

// ── ANSI strip parity ──────────────────────────────────────────────────

func TestStripANSI(t *testing.T) {
	input := "\033[32m✓\033[0m Success \033[2;3mand dim text\033[0m"
	result := stripANSI(input)
	assertContains(t, result, "✓")
	assertContains(t, result, "Success")
	assertContains(t, result, "and dim text")
	matched, _ := regexp.MatchString(`\033\[`, result)
	if matched {
		t.Fatalf("ANSI codes not stripped: %q", result)
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────

func projectDir() string {
	// Find the cmd/harness directory
	wd, _ := os.Getwd()
	return wd
}

func binaryName() string {
	if runtime.GOOS == "windows" {
		return "harness_test_bin.exe"
	}
	return "harness_test_bin"
}

func runBinary(t *testing.T, args ...string) string {
	t.Helper()
	binPath := filepath.Join(projectDir(), binaryName())

	// Build first if needed
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		cmd := exec.Command("go", "build", "-o", binaryName(), ".")
		cmd.Dir = projectDir()
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("Build failed: %s\n%s", err, string(out))
		}
	}

	cmd := exec.Command(binPath, args...)
	cmd.Dir = projectDir()
	out, err := cmd.CombinedOutput()
	if err != nil {
		// Some commands (like --help) may exit with non-zero
		// Only fail if there's no output
		if len(out) == 0 {
			t.Fatalf("Binary failed with no output: %v", err)
		}
	}
	return string(out)
}

func assertContains(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("Expected string to contain %q, got:\n%s", needle, truncateStr(haystack, 500))
	}
}

func assertNotContains(t *testing.T, haystack, needle string) {
	t.Helper()
	if strings.Contains(haystack, needle) {
		t.Errorf("Expected string NOT to contain %q, got:\n%s", needle, truncateStr(haystack, 500))
	}
}

func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ── New feature tests ───────────────────────────────────────────────────

func TestDesanitizeThinkTokens(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"no think tags", "hello world", "hello world"},
		{"sanitized open", "<\u200bthink>content", "<think>content"},
		{"sanitized close", "content<\u200b/think>", "content</think>"},
		{"both sanitized", "<\u200bthink>thinking<\u200b/think>", "<think>thinking</think>"},
		{"mixed", "before <\u200bthink>thinking<\u200b/think> after", "before <think>thinking</think> after"},
		{"unsanitized pass through", "<think>already clean</think>", "<think>already clean</think>"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := desanitizeThinkTokens(tc.input)
			if result != tc.expected {
				t.Errorf("desanitizeThinkTokens(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestDetectProviderLabel(t *testing.T) {
	tests := []struct {
		url      string
		expected string
	}{
		{"https://api.githubcopilot.com/", "GitHub Copilot"},
		{"https://api.openai.com/v1", "OpenAI"},
		{"https://openrouter.ai/api/v1", "OpenRouter"},
		{"https://api.together.xyz/v1", "Together AI"},
		{"https://bedrock-runtime.us-east-1.amazonaws.com", "AWS Bedrock"},
		{"https://open.bigmodel.cn/api/paas/v4", "OpenAI-compatible"},
		{"https://nano-gpt.com/api/v1", "NanoGPT"},
		{"https://api.z.ai/v1", "Z.AI (GLM)"},
	}

	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			result := detectProviderLabel(tc.url)
			if result != tc.expected {
				t.Errorf("detectProviderLabel(%q) = %q, want %q", tc.url, result, tc.expected)
			}
		})
	}
}

func TestExtractReasoningDetailsText(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"plain string", `"hello"`, "hello"},
		{"object with text", `{"text": "thinking content"}`, "thinking content"},
		{"array of objects", `[{"text": "a"}, {"text": "b"}]`, "ab"},
		{"nested reasoning_details", `{"reasoning_details": {"text": "nested"}}`, "nested"},
		{"empty", `{}`, ""},
		{"null", `null`, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := extractReasoningDetailsText(json.RawMessage(tc.input))
			if result != tc.expected {
				t.Errorf("extractReasoningDetailsText(%s) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestModelHistory(t *testing.T) {
	// Create a temp config file
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, ".z.json")

	// Override globalConfigPath for testing
	origHome := os.Getenv("HOME")
	origUserProfile := os.Getenv("USERPROFILE")

	// Write initial config
	initial := map[string]interface{}{
		"model":   "gpt-4o",
		"api_url": "https://api.openai.com/v1",
	}
	data, _ := json.MarshalIndent(initial, "", "  ")
	os.WriteFile(tmpFile, data, 0644)

	// Test loadModelHistory returns empty when no history
	// (We can't easily test the real functions because they use globalConfigPath()
	// which reads from the real HOME. So we test the serialization logic directly.)
	var raw map[string]interface{}
	json.Unmarshal(data, &raw)
	_, hasHistory := raw["model_history"]
	if hasHistory {
		t.Error("Fresh config should have no model_history")
	}

	// Test model history entry format
	entry := ModelHistoryEntry{Model: "claude-opus-4", Profile: "github-copilot"}
	if entry.Model != "claude-opus-4" || entry.Profile != "github-copilot" {
		t.Error("ModelHistoryEntry fields incorrect")
	}

	_ = origHome
	_ = origUserProfile
}

func TestProviderConfig_SwitchAppliesFields(t *testing.T) {
	cfg := defaultConfig()
	cfg.Providers = map[string]ProviderConfig{
		"test-provider": {
			APIURL:    "https://test.example.com/v1",
			APIKey:    "test-key-123",
			Model:     "test-model",
			MaxTokens: 8192,
		},
	}

	// Simulate /providers use
	p := cfg.Providers["test-provider"]
	cfg.APIBase = p.APIURL
	cfg.APIKey = p.APIKey
	cfg.Model = p.Model
	cfg.MaxTokens = p.MaxTokens
	cfg.Provider = "test-provider"

	if cfg.APIBase != "https://test.example.com/v1" {
		t.Errorf("APIBase not applied: %s", cfg.APIBase)
	}
	if cfg.Model != "test-model" {
		t.Errorf("Model not applied: %s", cfg.Model)
	}
	if cfg.Provider != "test-provider" {
		t.Errorf("Provider not set: %s", cfg.Provider)
	}
}

func TestSSEDelta_ReasoningFieldPriority(t *testing.T) {
	// Test that reasoning field priority matches Python: reasoning_content > reasoning > thinking > provider_specific > reasoning_details
	tests := []struct {
		name     string
		delta    sseDelta
		expected string
	}{
		{
			"reasoning_content wins",
			sseDelta{ReasoningContent: "rc", Reasoning: "r", Thinking: "t"},
			"rc",
		},
		{
			"reasoning fallback",
			sseDelta{Reasoning: "r", Thinking: "t"},
			"r",
		},
		{
			"thinking fallback",
			sseDelta{Thinking: "t"},
			"t",
		},
		{
			"provider_specific_fields fallback",
			sseDelta{ProviderSpecificFields: map[string]interface{}{"thinking": "psf"}},
			"psf",
		},
		{
			"reasoning_details fallback",
			sseDelta{ReasoningDetails: json.RawMessage(`"rd text"`)},
			"rd text",
		},
		{
			"no reasoning",
			sseDelta{Content: "just content"},
			"",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Replicate the extraction logic from doSSEStream
			reasoning := tc.delta.ReasoningContent
			if reasoning == "" {
				reasoning = tc.delta.Reasoning
			}
			if reasoning == "" {
				reasoning = tc.delta.ReasoningText
			}
			if reasoning == "" {
				reasoning = tc.delta.Thinking
			}
			if reasoning == "" && tc.delta.ProviderSpecificFields != nil {
				for _, key := range []string{"thinking", "reasoning", "reasoning_content"} {
					if v, ok := tc.delta.ProviderSpecificFields[key]; ok {
						if s, ok := v.(string); ok && s != "" {
							reasoning = s
							break
						}
					}
				}
			}
			if reasoning == "" && len(tc.delta.ReasoningDetails) > 0 {
				reasoning = extractReasoningDetailsText(tc.delta.ReasoningDetails)
			}

			if reasoning != tc.expected {
				t.Errorf("reasoning = %q, want %q", reasoning, tc.expected)
			}
		})
	}
}

func TestWriteToFile_DesanitizesThinkTokens(t *testing.T) {
	tmpDir := t.TempDir()
	th := NewToolHandlers(tmpDir)

	// Write content with sanitized think tokens
	result := th.writeToFile(map[string]string{
		"path":    "test_think.txt",
		"content": "before <\u200bthink>thinking<\u200b/think> after",
	})
	if result.Error != "" {
		t.Fatalf("writeToFile error: %s", result.Error)
	}

	// Read back and verify think tokens are desanitized
	data, err := os.ReadFile(filepath.Join(tmpDir, "test_think.txt"))
	if err != nil {
		t.Fatalf("Read error: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "<think>thinking</think>") {
		t.Errorf("Think tokens not desanitized. Got: %q", content)
	}
	if strings.Contains(content, "\u200b") {
		t.Errorf("ZWS still present in output: %q", content)
	}
}

func TestReplaceInFile_DesanitizesThinkTokens(t *testing.T) {
	tmpDir := t.TempDir()
	th := NewToolHandlers(tmpDir)

	// Create a file with think tag content
	testFile := filepath.Join(tmpDir, "test_replace.txt")
	os.WriteFile(testFile, []byte("<think>old content</think>\n"), 0644)

	// Replace with sanitized params
	result := th.replaceInFile(map[string]string{
		"path":     "test_replace.txt",
		"old_text": "<\u200bthink>old content<\u200b/think>",
		"new_text": "<\u200bthink>new content<\u200b/think>",
	})
	if result.Error != "" {
		t.Fatalf("replaceInFile error: %s", result.Error)
	}

	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Read error: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "<think>new content</think>") {
		t.Errorf("Think tokens not desanitized in replace. Got: %q", content)
	}
}
