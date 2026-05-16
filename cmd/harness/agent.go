package main

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Agent is the core agentic loop: chat → parse → tool → repeat.
type Agent struct {
	Workspace     string
	Config        *Config
	Session       *Session
	Client        *StreamingClient
	Tools         *ToolHandlers
	Status        *StatusLine
	Display       *Display
	Monitor       *KeyboardMonitor
	MaxIterations int
	SessionPath   string

	// Internal state
	hiddenOnlyCount int
	emptyNudgeCount int
	retryCount      int
	autoSaveEvery   int
	iterSinceLastSave int
}

// NewAgent creates and wires up all components.
func NewAgent(workspace string, cfg *Config, sessionPath string) *Agent {
	status := NewStatusLine()
	display := NewDisplay(status)
	monitor := NewKeyboardMonitor()

	client := NewStreamingClient(cfg)
	tools := NewToolHandlers(workspace)

	a := &Agent{
		Workspace:     workspace,
		Config:        cfg,
		Client:        client,
		Tools:         tools,
		Status:        status,
		Display:       display,
		Monitor:       monitor,
		MaxIterations: 500,
		SessionPath:   sessionPath,
		autoSaveEvery: 5,
	}
	return a
}

// SetSession sets or loads a session.
func (a *Agent) SetSession(sess *Session) {
	a.Session = sess
}

// RunMessage processes a user message through the agentic loop.
func (a *Agent) RunMessage(userInput string) string {
	if a.Session == nil {
		a.Session = NewSession(a.Workspace)
	}

	// Build system prompt if needed (first message, empty messages, or corrupt prompt)
	needsSysPrompt := len(a.Session.Messages) == 0 || a.Session.needsSystemPrompt
	if !needsSysPrompt && len(a.Session.Messages) > 0 && a.Session.Messages[0].Role != "system" {
		needsSysPrompt = true
	}
	if needsSysPrompt {
		idx := BuildWorkspaceIndex(a.Workspace)
		projectMap := idx.CompactTree()
		instructions := loadInstructionFiles(a.Workspace)
		sysPrompt := buildSystemPrompt(a.Workspace, projectMap, "", instructions)
		if len(a.Session.Messages) > 0 && a.Session.Messages[0].Role == "system" {
			// Replace existing corrupt system prompt
			a.Session.Messages[0] = Message{Role: "system", Content: sysPrompt}
		} else {
			// Insert system prompt at the beginning
			a.Session.Messages = append([]Message{{Role: "system", Content: sysPrompt}}, a.Session.Messages...)
		}
		a.Session.needsSystemPrompt = false
	}

	// Append user message
	a.Session.AppendMessage("user", userInput)

	// Start interrupt monitoring
	a.Monitor.Start()
	defer a.Monitor.Stop()

	// Run the loop
	result := a.runLoop()

	// Save session
	if a.SessionPath != "" {
		if err := SaveSession(a.Session, a.SessionPath); err != nil {
			logWarn("Failed to save session: %v", err)
		}
	}

	return result
}

// runLoop is the core iteration loop.
func (a *Agent) runLoop() string {
	a.hiddenOnlyCount = 0
	a.emptyNudgeCount = 0
	a.retryCount = 0

	for iteration := 1; iteration <= a.MaxIterations; iteration++ {
		logInfo("=== Iteration %d/%d ===", iteration, a.MaxIterations)
		a.Status.SetIteration(iteration, a.MaxIterations)
		a.Status.NewTurn()

		// Check interrupt
		if isInterrupted() {
			a.Status.Clear()
			if a.Display.IsTTY() {
				fmt.Println("\n\033[33m[STOP] Interrupted by user\033[0m")
			} else {
				fmt.Println("\n[STOP] Interrupted by user")
			}
			return "[Interrupted - session preserved. Type to continue or start new request]"
		}

		// Estimate tokens and compact if needed
		tokens := a.Session.EstimateTokens()
		maxTokens := a.Config.MaxContextTokens
		if maxTokens == 0 {
			maxTokens = 128000
		}
		a.Status.SetTokens(tokens, maxTokens)
		if tokens > int(float64(maxTokens)*0.85) {
			logInfo("Context at %d tokens (max %d), compacting...", tokens, maxTokens)
			a.compactContext()
			tokens = a.Session.EstimateTokens()
			a.Status.SetTokens(tokens, maxTokens)
		}

		// Make API call
		a.Status.Update("Sending to LLM", PhaseSending)
		a.Display.Reset()

		fullContent, reasoning, finishReason, err := a.callAPI()
		if iteration > 1 && reasoning == "" {
			logInfo("Iteration %d: NO reasoning tokens received from API (content_len=%d)", iteration, len(fullContent))
		}
		if err != nil {
			// Handle retriable errors
			if isRetriableError(err) {
				a.retryCount++
				if a.retryCount >= 5 {
					a.Status.Clear()
					return fmt.Sprintf("[API Error: %v (gave up after %d retries)]", err, a.retryCount)
				}
				a.Status.Update(fmt.Sprintf("Retrying (%d/5) after error...", a.retryCount), PhaseRetrying)
				logWarn("Retriable API error (%d/5): %v", a.retryCount, err)
				time.Sleep(5 * time.Second)
				continue
			}
			a.Status.Clear()
			return fmt.Sprintf("[API Error: %v]", err)
		}
		a.retryCount = 0 // Reset on success

		a.Status.Update("Processing response...", PhaseStreaming)

		// Check for interrupt during streaming
		if isInterrupted() {
			a.Status.Clear()
			if a.Display.IsTTY() {
				fmt.Println("\n\033[33m[STOP] Interrupted by user\033[0m")
			} else {
				fmt.Println("\n[STOP] Interrupted by user")
			}
			// Still save partial content
			if fullContent != "" {
				a.Session.AppendAssistant(fullContent+"\n[interrupted by user]", nil)
			}
			return "[Interrupted - session preserved. Type to continue or start new request]"
		}

		// Parse tool calls FIRST (before empty/hidden checks).
		// Python parses tools first — responses containing only thinking +
		// tool XML must not be treated as empty.
		displayText := StripThinkingTags(fullContent)
		displayText = StripToolTags(displayText)
		displayText = strings.TrimSpace(displayText)

		toolCalls := ParseToolCalls(fullContent)

		// If tool calls are present, skip empty/hidden checks entirely
		if len(toolCalls) > 0 {
			a.hiddenOnlyCount = 0
			a.emptyNudgeCount = 0
		} else {
			// No tool calls — check for empty/hidden-only responses
			if displayText == "" && len(reasoning) > 50 {
				// Hidden-only: reasoning but no visible content
				a.hiddenOnlyCount++
				if a.hiddenOnlyCount >= 2 {
					a.Session.AppendAssistant(fullContent, nil)
					a.Status.Clear()
					return "(Model produced only internal reasoning with no visible output)"
				}
				a.Session.AppendAssistant(fullContent, nil)
				a.Session.AppendMessage("user",
					"Your response contained only internal reasoning with no visible output. "+
						"Please proceed: either use a tool to continue the task, or use attempt_completion to present your final result.")
				logInfo("Hidden-only response %d/2, nudging", a.hiddenOnlyCount)
				continue
			}

			if fullContent == "" || (displayText == "" && reasoning == "") {
				// Truly empty
				a.emptyNudgeCount++
				if a.emptyNudgeCount >= 3 {
					a.Status.Clear()
					return "(Model returned empty responses repeatedly)"
				}
				a.Session.AppendAssistant(fullContent, nil)
				a.Session.AppendMessage("user",
					"Your previous response was incomplete or contained only internal thinking. "+
						"Please provide a clear, written response. Either use a tool to continue the task, "+
						"or use attempt_completion if the task is complete.")
				logInfo("Empty response %d/3, nudging", a.emptyNudgeCount)
				continue
			}

			// Reset empty counters on successful content
			a.hiddenOnlyCount = 0
			a.emptyNudgeCount = 0
		}

		// Check for truncated/unclosed tool tags
		if finishReason == "length" && hasUnclosedToolTag(fullContent) {
			logInfo("Truncated response with unclosed tool tag, requesting continuation")
			a.Session.AppendAssistant(fullContent, nil)
			a.Session.AppendMessage("user",
				"[SYSTEM: Your output was truncated before completing the tool call. "+
					"Please continue from where you left off. Do NOT repeat what you already wrote — "+
					"just output the remaining XML to complete the tool call.]")
			continue
		}

		if len(toolCalls) == 0 {
			// No tool calls — this is a final text response
			a.Session.AppendAssistant(fullContent, nil)
			a.Display.RenderFinalContent(fullContent)
			a.Status.Clear()
			return fullContent
		}

		// Execute first tool (1-tool-per-turn policy)
		tc := toolCalls[0]

		// Check for attempt_completion
		if tc.Name == "attempt_completion" {
			a.Session.AppendAssistant(fullContent, nil)
			result := a.Tools.Dispatch(tc)
			// Render the result parameter content (not pre-tool text)
			completionText := result.Output
			if completionText == "" {
				completionText = displayText
			}
			a.Display.RenderFinalContent(completionText)
			a.Status.Clear()
			return completionText
		}

		// Execute the tool
		a.Status.Update(fmt.Sprintf("Executing: %s", tc.Name), PhaseToolExec)
		logInfo("Tool: %s", tc.Name)

		result := a.Tools.Dispatch(tc)

		// Don't render pre-tool text on tool turns (matches Python behavior).
		// The model's commentary before tool XML is suppressed — only tool
		// results are shown.

		a.Session.AppendAssistant(fullContent, nil)

		// Build tool result message
		var toolMsg strings.Builder

		// Inject active todos for grounding
		todoSummary := a.getTodoSummary()
		if todoSummary != "" {
			toolMsg.WriteString("[ACTIVE TODOS]\n")
			toolMsg.WriteString(todoSummary)
			toolMsg.WriteString("\n\n")
		}

		toolMsg.WriteString(fmt.Sprintf("[%s result]\n", tc.Name))
		if result.Error != "" {
			toolMsg.WriteString(fmt.Sprintf("Error: %s\n", result.Error))
		}
		if result.Output != "" {
			toolMsg.WriteString(result.Output)
		}

		// Report extra tool calls that were not executed
		if len(toolCalls) > 1 {
			extra := make([]string, 0, len(toolCalls)-1)
			for _, etc := range toolCalls[1:] {
				extra = append(extra, etc.Name)
			}
			toolMsg.WriteString(fmt.Sprintf(
				"\n\n[SYSTEM: Only 1 tool call is executed per turn. The following %d tool call(s) were NOT executed: %s. Please re-issue them one at a time in subsequent turns.]",
				len(extra), strings.Join(extra, ", ")))
		}

		a.Session.AppendMessage("user", toolMsg.String())

		// Output tool result to terminal
		a.showToolResult(tc.Name, tc.Params, result)

		// Auto-save
		a.iterSinceLastSave++
		if a.iterSinceLastSave >= a.autoSaveEvery && a.SessionPath != "" {
			a.iterSinceLastSave = 0
			if err := SaveSession(a.Session, a.SessionPath); err != nil {
				logWarn("Auto-save failed: %v", err)
			}
		}

		// Check interrupt after tool execution
		if isInterrupted() {
			a.Status.Clear()
			if a.Display.IsTTY() {
				fmt.Println("\n\033[33m[STOP] Interrupted by user\033[0m")
			} else {
				fmt.Println("\n[STOP] Interrupted by user")
			}
			return "[Interrupted - session preserved. Type to continue or start new request]"
		}
	}

	a.Status.Clear()
	return "Max iterations reached."
}

// callAPI makes a streaming API call and returns the full content, reasoning, and finish reason.
func (a *Agent) callAPI() (fullContent string, reasoning string, finishReason string, err error) {
	// Convert session messages to API format
	msgs := make([]map[string]interface{}, 0, len(a.Session.Messages))
	for _, m := range a.Session.Messages {
		msgs = append(msgs, map[string]interface{}{
			"role":    m.Role,
			"content": m.Content,
		})
	}

	var contentBuf strings.Builder
	var reasoningBuf strings.Builder

	a.Display.Reset()
	a.Status.Update("Streaming...", PhaseStreaming)

	onChunk := func(chunk StreamChunk) {
		if chunk.Content != "" {
			contentBuf.WriteString(chunk.Content)
			// Process through display filter for thinking tag extraction only.
			// Visible content is NOT printed during streaming — it will be
			// rendered once after streaming completes (deferred mode, like Python).
			a.Display.OnChunk(chunk.Content)
		}
		if chunk.Reasoning != "" {
			reasoningBuf.WriteString(chunk.Reasoning)
			a.Display.OnReasoning(chunk.Reasoning)
		}
		if chunk.FinishReason != "" {
			finishReason = chunk.FinishReason
		}
	}

	err = a.Client.ChatStream(msgs, onChunk)
	if err != nil {
		return "", "", "", err
	}

	a.Display.FlushThinking()

	logInfo("callAPI done: reasoning_len=%d content_len=%d finish=%s", reasoningBuf.Len(), contentBuf.Len(), finishReason)

	return contentBuf.String(), reasoningBuf.String(), finishReason, nil
}

// hasUnclosedToolTag checks if any tool tag was opened but never closed.
func hasUnclosedToolTag(content string) bool {
	for name := range toolTagNames {
		openTag := "<" + name + ">"
		openTagSpace := "<" + name + " "
		closeTag := "</" + name + ">"
		if (strings.Contains(content, openTag) || strings.Contains(content, openTagSpace)) &&
			!strings.Contains(content, closeTag) {
			return true
		}
	}
	return false
}

// isRetriableError checks if an API error is retriable (rate limit, transient, etc.)
func isRetriableError(err error) bool {
	msg := err.Error()
	retriable := []string{"429", "rate limit", "throttl", "timeout", "connection reset", "temporary", "quota"}
	for _, r := range retriable {
		if strings.Contains(strings.ToLower(msg), r) {
			return true
		}
	}
	return false
}

// compactContext performs a simple context compaction.
func (a *Agent) compactContext() {
	a.Status.Update("Compacting context...", PhaseCompacting)
	logInfo("Starting context compaction")

	if len(a.Session.Messages) < 4 {
		a.Status.Clear()
		return
	}

	// Keep: system prompt (0), first user message (1), last 6 messages
	keep := 6
	if len(a.Session.Messages) <= keep+2 {
		a.Status.Clear()
		return
	}

	// Build compaction summary
	var summary strings.Builder
	summary.WriteString("[CONTEXT COMPACTED]\nEarlier conversation was compacted to save context space.\n\n")

	// Summarize removed messages
	removed := a.Session.Messages[2 : len(a.Session.Messages)-keep]
	toolCounts := make(map[string]int)
	for _, m := range removed {
		content := MessageContent(m)
		calls := ParseToolCalls(content)
		for _, tc := range calls {
			toolCounts[tc.Name]++
		}
	}
	if len(toolCounts) > 0 {
		summary.WriteString("Tools used in compacted context: ")
		parts := make([]string, 0, len(toolCounts))
		for name, count := range toolCounts {
			parts = append(parts, fmt.Sprintf("%s(%d)", name, count))
		}
		summary.WriteString(strings.Join(parts, ", "))
		summary.WriteString("\n")
	}

	// Rebuild messages: system + first user + compaction notice + last N
	newMsgs := make([]Message, 0, keep+3)
	newMsgs = append(newMsgs, a.Session.Messages[0]) // system
	newMsgs = append(newMsgs, a.Session.Messages[1]) // first user
	newMsgs = append(newMsgs, Message{Role: "user", Content: summary.String()})
	newMsgs = append(newMsgs, a.Session.Messages[len(a.Session.Messages)-keep:]...)

	a.Session.Messages = newMsgs
	a.Status.Clear()
	logInfo("Compacted: %d messages removed, %d remaining", len(removed), len(newMsgs))
}

// getTodoSummary returns a brief summary of active todos.
func (a *Agent) getTodoSummary() string {
	// TODO: integrate with todo manager
	return ""
}

// showToolResult displays tool execution result to the user.
func (a *Agent) showToolResult(toolName string, params map[string]string, result ToolResult) {
	a.Status.PrintSafe(func() {
		tty := a.Display.IsTTY()
		if result.Error != "" {
			errLine := firstLine(result.Error)
			if tty {
				fmt.Fprintf(os.Stderr, "\033[31m  ✗ %s: %s\033[0m\n", toolName, truncate(errLine, 120))
			} else {
				fmt.Fprintf(os.Stderr, "  ✗ %s: %s\n", toolName, truncate(errLine, 120))
			}
		} else {
			summary := toolResultSummary(toolName, params, result.Output)
			if tty {
				fmt.Fprintf(os.Stderr, "\033[32m  ✓ %s\033[0m %s\n", toolName, summary)
			} else {
				fmt.Fprintf(os.Stderr, "  ✓ %s %s\n", toolName, summary)
			}
		}
	})
}

// toolResultSummary produces a clean one-line summary of a tool's output.
func toolResultSummary(toolName string, params map[string]string, output string) string {
	if output == "" {
		return ""
	}
	switch toolName {
	case "read_file":
		// Show filename and line range
		path := params["path"]
		if s, ok := params["start_line"]; ok && s != "" {
			if e, ok := params["end_line"]; ok && e != "" {
				return fmt.Sprintf("%s (%s-%s)", path, s, e)
			}
			return fmt.Sprintf("%s (%s-EOF)", path, s)
		}
		// Count lines returned
		n := strings.Count(output, "\n")
		return fmt.Sprintf("%s (%d lines)", path, n)
	case "write_to_file":
		return firstLine(output)
	case "replace_in_file":
		path := params["path"]
		if path != "" {
			return path
		}
		return firstLine(output)
	case "execute_command":
		cmd := params["command"]
		if cmd != "" {
			return truncate(cmd, 60)
		}
		line := firstLine(output)
		if line == "" {
			return "(no output)"
		}
		return truncate(line, 80)
	case "list_files":
		path := params["path"]
		n := strings.Count(strings.TrimSpace(output), "\n") + 1
		if path != "" {
			return fmt.Sprintf("%s (%d entries)", path, n)
		}
		return fmt.Sprintf("%d entries", n)
	case "search_files":
		pattern := params["regex"]
		path := params["path"]
		n := strings.Count(output, "\n")
		if pattern != "" && path != "" {
			return fmt.Sprintf("\"%s\" in %s (%d matches)", truncate(pattern, 30), path, n)
		}
		if pattern != "" {
			return fmt.Sprintf("\"%s\" (%d matches)", truncate(pattern, 30), n)
		}
		return truncate(firstLine(output), 80)
	default:
		return truncate(firstLine(output), 80)
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
