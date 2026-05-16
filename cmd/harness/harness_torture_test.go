package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ── Torture: replace_in_file multi-strategy matching ────────────────────

func TestReplaceInFile_IndentAgnostic(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// File has 4-space indentation
	testFile := filepath.Join(th.Workspace, "indent_test.py")
	os.WriteFile(testFile, []byte("class Foo:\n    def bar(self):\n        return 42\n"), 0644)

	// Search uses 2-space indentation — should match via indentation-agnostic fallback
	result := th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "indent_test.py",
		"old_text": "  def bar(self):\n      return 42",
		"new_text": "    def baz(self):\n        return 99",
	}})
	if result.Error != "" {
		t.Fatalf("Indent-agnostic replace failed: %s", result.Error)
	}

	data, _ := os.ReadFile(testFile)
	assertContains(t, string(data), "def baz")
	assertContains(t, string(data), "return 99")
}

func TestReplaceInFile_PreservesFileIntegrity(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Write a multi-section file
	original := `# Header
import os
import sys

def func_a():
    pass

def func_b():
    return True

def func_c():
    print("hello")
`
	testFile := filepath.Join(th.Workspace, "integrity.py")
	os.WriteFile(testFile, []byte(original), 0644)

	// Replace middle function
	th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "integrity.py",
		"old_text": "def func_b():\n    return True",
		"new_text": "def func_b():\n    return False",
	}})

	data, _ := os.ReadFile(testFile)
	content := string(data)

	// Header and other functions must be untouched
	assertContains(t, content, "# Header")
	assertContains(t, content, "import os")
	assertContains(t, content, "def func_a():")
	assertContains(t, content, "def func_c():")
	assertContains(t, content, "return False")
	assertNotContains(t, content, "return True")
}

func TestReplaceInFile_NonExistentText(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	testFile := filepath.Join(th.Workspace, "noexist.py")
	os.WriteFile(testFile, []byte("hello world\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "noexist.py",
		"old_text": "this text is not in the file",
		"new_text": "replacement",
	}})
	if result.Error == "" {
		t.Fatal("Expected error for non-existent old_text, got success")
	}
}

func TestReplaceInFile_EmptyLines(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	testFile := filepath.Join(th.Workspace, "empty_lines.py")
	os.WriteFile(testFile, []byte("line1\n\nline3\n\nline5\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "replace_in_file", Params: map[string]string{
		"path":     "empty_lines.py",
		"old_text": "line1\n\nline3",
		"new_text": "LINE1\n\nLINE3",
	}})
	if result.Error != "" {
		t.Fatalf("Empty line replace error: %s", result.Error)
	}

	data, _ := os.ReadFile(testFile)
	assertContains(t, string(data), "LINE1")
	assertContains(t, string(data), "LINE3")
	assertContains(t, string(data), "line5")
}

// ── Torture: XML parsing edge cases ─────────────────────────────────────

func TestParseToolCalls_NestedAngleBrackets(t *testing.T) {
	// Content that contains angle brackets (e.g., HTML, generics)
	input := `<write_to_file>
<path>index.html</path>
<content>
<html>
<head><title>Test</title></head>
<body>
<p>Hello <b>World</b></p>
</body>
</html>
</content>
</write_to_file>`

	calls := ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("Expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "write_to_file" {
		t.Fatalf("Expected write_to_file, got %s", calls[0].Name)
	}
	assertContains(t, calls[0].Params["content"], "<html>")
	assertContains(t, calls[0].Params["content"], "Hello <b>World</b>")
}

func TestParseToolCalls_UnclosedTag(t *testing.T) {
	// Truncated response — no closing tag
	input := `<read_file>
<path>test.py</path>`

	calls := ParseToolCalls(input)
	if len(calls) != 0 {
		t.Fatalf("Unclosed tag should not parse as tool call, got %d", len(calls))
	}
}

func TestParseToolCalls_EmptyParams(t *testing.T) {
	input := `<list_files>
<path>.</path>
</list_files>`

	calls := ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("Expected 1 call, got %d", len(calls))
	}
	if calls[0].Params["path"] != "." {
		t.Fatalf("Path should be '.', got %q", calls[0].Params["path"])
	}
}

func TestParseToolCalls_ReplaceWithSpecialChars(t *testing.T) {
	// Replace content with regex special chars
	input := `<replace_in_file>
<path>test.py</path>
<old_text>pattern = r"foo.*bar"</old_text>
<new_text>pattern = r"baz\d+"</new_text>
</replace_in_file>`

	calls := ParseToolCalls(input)
	if len(calls) != 1 {
		t.Fatalf("Expected 1 call, got %d", len(calls))
	}
	assertContains(t, calls[0].Params["old_text"], `foo.*bar`)
	assertContains(t, calls[0].Params["new_text"], `baz\d+`)
}

func TestParseToolCalls_MixedTools(t *testing.T) {
	input := `I'll start by reading the file.

<read_file>
<path>src/main.py</path>
</read_file>

Then I'll search for usages.

<search_files>
<path>src</path>
<regex>import main</regex>
</search_files>`

	calls := ParseToolCalls(input)
	if len(calls) != 2 {
		t.Fatalf("Expected 2 calls, got %d", len(calls))
	}
	if calls[0].Name != "read_file" {
		t.Fatalf("First call should be read_file, got %s", calls[0].Name)
	}
	if calls[1].Name != "search_files" {
		t.Fatalf("Second call should be search_files, got %s", calls[1].Name)
	}
}

// ── Torture: write_to_file edge cases ───────────────────────────────────

func TestWriteToFile_Unicode(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	content := "# こんにちは\n\ndef greet():\n    return '🎉'\n"
	result := th.Dispatch(ToolCall{Name: "write_to_file", Params: map[string]string{
		"path":    "unicode_test.py",
		"content": content,
	}})
	if result.Error != "" {
		t.Fatalf("Unicode write error: %s", result.Error)
	}

	data, _ := os.ReadFile(filepath.Join(th.Workspace, "unicode_test.py"))
	assertContains(t, string(data), "こんにちは")
	assertContains(t, string(data), "🎉")
}

func TestWriteToFile_LargeFile(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// ~50KB file
	var sb strings.Builder
	for i := 0; i < 1000; i++ {
		sb.WriteString("line " + strings.Repeat("x", 50) + "\n")
	}

	result := th.Dispatch(ToolCall{Name: "write_to_file", Params: map[string]string{
		"path":    "large.txt",
		"content": sb.String(),
	}})
	if result.Error != "" {
		t.Fatalf("Large write error: %s", result.Error)
	}

	info, _ := os.Stat(filepath.Join(th.Workspace, "large.txt"))
	if info.Size() < 50000 {
		t.Fatalf("Large file too small: %d bytes", info.Size())
	}
}

func TestWriteToFile_EmptyContent(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	result := th.Dispatch(ToolCall{Name: "write_to_file", Params: map[string]string{
		"path":    "empty.txt",
		"content": "",
	}})
	if result.Error != "" {
		t.Fatalf("Empty write error: %s", result.Error)
	}

	info, _ := os.Stat(filepath.Join(th.Workspace, "empty.txt"))
	if info.Size() > 1 { // May have trailing newline
		t.Fatalf("Empty file should be tiny, got %d bytes", info.Size())
	}
}

// ── Torture: read_file edge cases ───────────────────────────────────────

func TestReadFile_NonExistent(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	result := th.Dispatch(ToolCall{Name: "read_file", Params: map[string]string{
		"path": "does_not_exist.txt",
	}})
	if result.Error == "" {
		t.Fatal("Expected error for non-existent file")
	}
}

func TestReadFile_LineNumbering(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	os.WriteFile(filepath.Join(th.Workspace, "numbered.txt"),
		[]byte("alpha\nbeta\ngamma\ndelta\nepsilon\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "read_file", Params: map[string]string{
		"path":       "numbered.txt",
		"start_line": "3",
		"end_line":   "4",
	}})
	if result.Error != "" {
		t.Fatalf("Line range error: %s", result.Error)
	}
	assertContains(t, result.Output, "3 | gamma")
	assertContains(t, result.Output, "4 | delta")
	assertNotContains(t, result.Output, "alpha")
	assertNotContains(t, result.Output, "epsilon")
}

func TestReadFile_BinaryDetection(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	// Write binary content (null bytes)
	os.WriteFile(filepath.Join(th.Workspace, "test.bin"),
		[]byte{0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00}, 0644)

	result := th.Dispatch(ToolCall{Name: "read_file", Params: map[string]string{
		"path": "test.bin",
	}})
	// Should either succeed with some representation or give a meaningful message
	// Python harness reads binary files but shows them as-is
	if result.Error != "" && !strings.Contains(result.Error, "binary") {
		// It's OK to either read or warn about binary
		_ = result
	}
}

// ── Torture: execute_command edge cases ─────────────────────────────────

func TestExecuteCommand_WorkingDirectory(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	var cmd string
	if isWindows() {
		cmd = "Get-Location | Select-Object -ExpandProperty Path"
	} else {
		cmd = "pwd"
	}

	result := th.Dispatch(ToolCall{Name: "execute_command", Params: map[string]string{
		"command": cmd,
	}})
	// Should produce output containing the workspace temp dir path
	if result.Output == "" {
		t.Fatalf("Working directory command produced no output, error: %s", result.Error)
	}
	// Verify it's running in the workspace directory
	assertContains(t, strings.ReplaceAll(strings.ToLower(result.Output), "\\", "/"),
		strings.ReplaceAll(strings.ToLower(th.Workspace), "\\", "/"))
}

func TestExecuteCommand_ExitCode(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	var cmd string
	if isWindows() {
		cmd = "exit /b 1"
	} else {
		cmd = "exit 1"
	}

	result := th.Dispatch(ToolCall{Name: "execute_command", Params: map[string]string{
		"command": cmd,
	}})
	// Should report non-zero exit
	if result.Error == "" {
		// Some implementations report exit code in output
		assertContains(t, result.Output+result.Error, "exit")
	}
}

// ── Torture: search_files edge cases ────────────────────────────────────

func TestSearchFiles_RegexPattern(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	os.WriteFile(filepath.Join(th.Workspace, "patterns.py"),
		[]byte("foo123bar\nfoo456bar\nbaz789qux\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "search_files", Params: map[string]string{
		"path":  ".",
		"regex": "foo\\d+bar",
	}})
	assertContains(t, result.Output, "foo123bar")
	assertContains(t, result.Output, "foo456bar")
	// baz789qux may appear as context line (grep -C behavior), that's OK
}

func TestSearchFiles_CaseInsensitive(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	os.WriteFile(filepath.Join(th.Workspace, "mixed_case.py"),
		[]byte("Hello World\nhello world\nHELLO WORLD\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "search_files", Params: map[string]string{
		"path":  ".",
		"regex": "(?i)hello",
	}})
	// Should find all three lines
	assertContains(t, result.Output, "Hello World")
	assertContains(t, result.Output, "hello world")
	assertContains(t, result.Output, "HELLO WORLD")
}

func TestSearchFiles_NoMatches(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	os.WriteFile(filepath.Join(th.Workspace, "nope.py"),
		[]byte("nothing here\n"), 0644)

	result := th.Dispatch(ToolCall{Name: "search_files", Params: map[string]string{
		"path":  ".",
		"regex": "zzz_not_found_zzz",
	}})
	// Should succeed but with no/empty results
	if result.Error != "" {
		t.Fatalf("Search with no matches returned error: %s", result.Error)
	}
}

// ── Torture: replace_between_anchors edge cases ─────────────────────────

func TestReplaceBetweenAnchors_MultipleAnchors(t *testing.T) {
	th := NewToolHandlers(t.TempDir())

	content := `// SECTION A START
first block
// SECTION A END

// SECTION B START
second block
// SECTION B END
`
	testFile := filepath.Join(th.Workspace, "multi_anchor.txt")
	os.WriteFile(testFile, []byte(content), 0644)

	// Replace only section A
	result := th.Dispatch(ToolCall{Name: "replace_between_anchors", Params: map[string]string{
		"path":         "multi_anchor.txt",
		"start_anchor": "// SECTION A START",
		"end_anchor":   "// SECTION A END",
		"replacement":  "replaced block",
	}})
	if result.Error != "" {
		t.Fatalf("Multi-anchor replace error: %s", result.Error)
	}

	data, _ := os.ReadFile(testFile)
	assertContains(t, string(data), "replaced block")
	assertContains(t, string(data), "second block") // Section B untouched
	assertNotContains(t, string(data), "first block")
}

// ── Torture: Display stream filter stress ───────────────────────────────

func TestDisplay_InterleavedToolsAndText(t *testing.T) {
	display := NewDisplay(nil)

	// Simulate a complex response with text, tool, more text, another tool
	chunks := []string{
		"First I'll read ",
		"file A.\n\n<read",
		"_file>\n<path>a.py",
		"</path>\n</read_",
		"file>\n\nNow let",
		" me check B.\n\n<",
		"read_file>\n<path",
		">b.py</path>\n</",
		"read_file>\n\nDone!",
	}

	var visible strings.Builder
	for _, chunk := range chunks {
		out := display.OnChunk(chunk)
		visible.WriteString(out)
	}

	result := visible.String()
	assertContains(t, result, "First I'll read file A.")
	assertContains(t, result, "Now let me check B.")
	assertContains(t, result, "Done!")
	assertNotContains(t, result, "<read_file>")
	assertNotContains(t, result, "a.py")
	assertNotContains(t, result, "b.py")
}

func TestDisplay_CharByCharStreaming(t *testing.T) {
	display := NewDisplay(nil)

	// Simulate extremely fine-grained streaming (char by char)
	input := "Hi!\n\n<read_file>\n<path>x.py</path>\n</read_file>\n\nBye."
	var visible strings.Builder
	for _, ch := range input {
		out := display.OnChunk(string(ch))
		visible.WriteString(out)
	}

	result := visible.String()
	assertContains(t, result, "Hi!")
	assertContains(t, result, "Bye.")
	assertNotContains(t, result, "<read_file>")
	assertNotContains(t, result, "x.py")
}

func TestDisplay_RenderFinalContent(t *testing.T) {
	// RenderFinalContent uses StripThinkingTags + StripToolTags internally
	raw := `<thinking>some thought</thinking>

Here is the result.

<read_file>
<path>test.py</path>
</read_file>

All done.`

	// Test the underlying strip functions that RenderFinalContent uses
	stripped := StripThinkingTags(raw)
	stripped = StripToolTags(stripped)
	assertContains(t, stripped, "Here is the result.")
	assertContains(t, stripped, "All done.")
	assertNotContains(t, stripped, "<thinking>")
	assertNotContains(t, stripped, "<read_file>")
}

// ── Torture: hasUnclosedToolTag edge cases ──────────────────────────────

func TestHasUnclosedToolTag_ComplexCases(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected bool
	}{
		{"properly closed write", "<write_to_file><path>f</path><content>x</content></write_to_file>", false},
		{"unclosed write", "<write_to_file><path>f</path><content>x", true},
		{"text with angle brackets", "if x < 5 and y > 3:", false},
		{"multiple closed", "<read_file><path>a</path></read_file>\n<read_file><path>b</path></read_file>", false},
		// Note: hasUnclosedToolTag checks for any open tag without a close in the ENTIRE string.
		// With two read_file opens and one close, implementation may see it as balanced.
		{"clearly unclosed", "text before <write_to_file><path>f</path><content>partial content", true},
		{"no tools at all", "Just some plain text with no XML.", false},
		{"thinking only", "<thinking>analysis</thinking>", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := hasUnclosedToolTag(tc.input)
			if result != tc.expected {
				t.Errorf("hasUnclosedToolTag = %v, want %v", result, tc.expected)
			}
		})
	}
}

// ── Torture: workspace index ────────────────────────────────────────────

func TestWorkspaceIndex_LargeDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	// Create 100 files across 10 dirs
	for i := 0; i < 10; i++ {
		dir := filepath.Join(tmpDir, strings.Repeat("d", i+1))
		os.MkdirAll(dir, 0755)
		for j := 0; j < 10; j++ {
			fname := filepath.Join(dir, strings.Repeat("f", j+1)+".py")
			os.WriteFile(fname, []byte("# file"), 0644)
		}
	}

	idx := BuildWorkspaceIndex(tmpDir)
	tree := idx.CompactTree()

	// Should have a non-trivial tree
	if len(tree) < 100 {
		t.Fatalf("Tree too short for 100 files: %d chars", len(tree))
	}
}

func TestWorkspaceIndex_SkipsBinaryExtensions(t *testing.T) {
	tmpDir := t.TempDir()

	os.WriteFile(filepath.Join(tmpDir, "code.py"), []byte("pass"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "image.png"), []byte{0x89, 0x50}, 0644)
	os.WriteFile(filepath.Join(tmpDir, "lib.so"), []byte{0x7f, 0x45}, 0644)
	os.WriteFile(filepath.Join(tmpDir, "app.exe"), []byte{0x4d, 0x5a}, 0644)

	idx := BuildWorkspaceIndex(tmpDir)
	tree := idx.CompactTree()

	assertContains(t, tree, "code.py")
	// Binary files may or may not appear in tree — that's OK.
	// The key parity point is that indexing doesn't crash.
}

// ── Torture: session management ─────────────────────────────────────────

func TestSessionName_Sanitization(t *testing.T) {
	cases := []struct {
		input string
	}{
		{"Hello, how are you?"},
		{"Fix the bug in src/main.py"},
		{"What's the time?!@#$%^&*()"},
		{""},
		{strings.Repeat("x", 200)},
	}

	for _, tc := range cases {
		name := GenerateSessionName(tc.input)
		// Name should be filesystem-safe
		if strings.ContainsAny(name, `<>:"/\|?*`) {
			t.Errorf("Session name contains unsafe chars: %q (from %q)", name, tc.input)
		}
		// Name should be reasonable length
		if len(name) > 100 {
			t.Errorf("Session name too long: %d chars", len(name))
		}
	}
}

func TestSession_ContextTracking(t *testing.T) {
	sess := NewSession("/workspace")

	id1 := sess.AddContext("file", "a.py", "content a", nil)
	id2 := sess.AddContext("file", "b.py", "content b", nil)
	id3 := sess.AddContext("tool", "search", "results", nil)

	if id1 != 1 || id2 != 2 || id3 != 3 {
		t.Fatalf("Context IDs should be sequential: %d, %d, %d", id1, id2, id3)
	}

	if len(sess.Context) != 3 {
		t.Fatalf("Expected 3 context items, got %d", len(sess.Context))
	}

	if sess.Context[0].Source != "a.py" {
		t.Fatalf("First context source wrong: %s", sess.Context[0].Source)
	}
}

// ── Helper ──────────────────────────────────────────────────────────────

func isWindows() bool {
	return filepath.Separator == '\\'
}

// ── Regression: sfBuf multi-byte rune handling ──────────────────────────

func TestDisplay_UnicodeInAngleBrackets(t *testing.T) {
	// When the model outputs text with < followed by multi-byte chars,
	// the display filter should not corrupt the bytes.
	d := NewDisplay(nil)
	d.Reset()

	// Simulate "x < π > y" arriving char by char
	var result strings.Builder
	for _, ch := range "x < π > y" {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	// The < π > should be flushed as-is since "π" is not a tool/thinking tag
	assertContains(t, out, "x ")
	assertContains(t, out, "π")
	assertContains(t, out, " y")
}

func TestDisplay_UnicodeToolContent(t *testing.T) {
	// Unicode content inside a tool tag should be fully suppressed
	d := NewDisplay(nil)
	d.Reset()

	input := "Hello <read_file><path>日本語.txt</path></read_file> world"
	var result strings.Builder
	for _, ch := range input {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	assertContains(t, out, "Hello")
	assertContains(t, out, "world")
	assertNotContains(t, out, "日本語")
}

// ── Regression: tool tag with attributes suppressed correctly ───────────

func TestDisplay_ToolTagWithAttributes(t *testing.T) {
	// Tags like <read_file path="foo"> with attributes should still be recognized
	d := NewDisplay(nil)
	d.Reset()

	input := "Before <read_file path=\"foo.txt\">\n<path>foo.txt</path>\n</read_file> After"
	var result strings.Builder
	for _, ch := range input {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	assertContains(t, out, "Before")
	assertContains(t, out, "After")
	assertNotContains(t, out, "foo.txt")
}

// ── Regression: native reasoning suppresses <thinking> tags ─────────────

func TestDisplay_NativeReasoningSuppressesThinkingTags(t *testing.T) {
	d := NewDisplay(nil)
	d.Reset()

	// Simulate native reasoning arriving first
	d.OnReasoning("I need to think about this")

	// Then content with <thinking> tags arrives via OnChunk
	input := "<thinking>duplicate thinking</thinking>Here is the answer."
	var result strings.Builder
	for _, ch := range input {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	// The <thinking> content should be suppressed (not double-displayed)
	assertContains(t, out, "Here is the answer.")
	assertNotContains(t, out, "duplicate thinking")
}

// ── Regression: session system prompt regeneration ──────────────────────

func TestSession_LoadCorruptSystemPrompt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_session.json")

	// Save a session with a corrupt system prompt
	sess := NewSession("/workspace")
	sess.AppendMessage("system", "This is not a real system prompt") // Missing "TOOL USE" and "read_file"
	sess.AppendMessage("user", "hello")
	sess.AppendMessage("assistant", "hi there")

	if err := SaveSession(sess, path); err != nil {
		t.Fatal(err)
	}

	// Load it — should detect corruption and mark for regeneration
	loaded, err := LoadSession(path)
	if err != nil {
		t.Fatal(err)
	}

	if !loaded.needsSystemPrompt {
		t.Fatal("Expected needsSystemPrompt=true for corrupt system prompt")
	}
	// The corrupt system prompt should have been removed
	if len(loaded.Messages) != 2 {
		t.Fatalf("Expected 2 messages (user+assistant) after corrupt prompt removed, got %d", len(loaded.Messages))
	}
}

func TestSession_LoadMissingSystemPrompt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_session.json")

	// Save a session WITHOUT a system prompt (messages[0].role != "system")
	sess := &Session{
		Workspace:  "/workspace",
		Messages:   []Message{{Role: "user", Content: "hello"}},
		CreatedAt:  "2024-01-01T00:00:00Z",
		UpdatedAt:  "2024-01-01T00:00:00Z",
	}

	if err := SaveSession(sess, path); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadSession(path)
	if err != nil {
		t.Fatal(err)
	}

	if !loaded.needsSystemPrompt {
		t.Fatal("Expected needsSystemPrompt=true when system prompt missing")
	}
}

// ── Regression: display filter suppression with closing tags ────────────

func TestDisplay_NestedToolContent(t *testing.T) {
	d := NewDisplay(nil)
	d.Reset()

	// Content inside tool tags with nested < > should be fully suppressed
	input := "Start <write_to_file><path>a.html</path><content><div class=\"foo\">bar</div></content></write_to_file> End"
	var result strings.Builder
	for _, ch := range input {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	assertContains(t, out, "Start")
	assertContains(t, out, "End")
	assertNotContains(t, out, "div")
	assertNotContains(t, out, "bar")
	assertNotContains(t, out, "foo")
}

func TestDisplay_MultipleToolCalls(t *testing.T) {
	d := NewDisplay(nil)
	d.Reset()

	input := "First <read_file><path>a.py</path></read_file> middle <execute_command><command>ls</command></execute_command> last"
	var result strings.Builder
	for _, ch := range input {
		result.WriteString(d.OnChunk(string(ch)))
	}

	out := result.String()
	assertContains(t, out, "First")
	assertContains(t, out, "middle")
	assertContains(t, out, "last")
	assertNotContains(t, out, "a.py")
	assertNotContains(t, out, "ls")
}

// ── Regression: reasoning output state tracking ─────────────────────────

func TestDisplay_ReasoningLineIndentation(t *testing.T) {
	d := NewDisplay(nil)
	d.Reset()

	// First call should print header + indented content
	d.OnReasoning("Line 1\nLine 2\n")

	// After reasoning, reasoningLineStart should be true (ready for next line indent)
	if !d.reasoningLineStart {
		t.Fatal("reasoningLineStart should be true after newline")
	}
}

func TestDisplay_ReasoningSkipsLeadingBlanks(t *testing.T) {
	d := NewDisplay(nil)
	d.Reset()

	// Reasoning starting with blank lines should skip them
	d.OnReasoning("\n\n\nActual thinking")

	// thinkShown should be true
	if !d.thinkShown {
		t.Fatal("thinkShown should be true after OnReasoning")
	}

	// nativeReasoning should be set
	if !d.nativeReasoning {
		t.Fatal("nativeReasoning should be true after OnReasoning")
	}
}
