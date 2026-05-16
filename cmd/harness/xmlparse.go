package main

import (
	"fmt"
	"regexp"
	"strings"
)

// ToolCall represents a parsed XML tool invocation.
type ToolCall struct {
	Name   string
	Params map[string]string
	Raw    string // The full XML match
}

// complexTools use greedy matching (last close tag wins).
var complexTools = map[string]bool{
	"write_to_file":          true,
	"replace_in_file":        true,
	"replace_between_anchors": true,
	"mcp_call_tool":          true,
}

// allToolNames is the complete list of supported tools.
var allToolNames = []string{
	"read_file", "write_to_file", "replace_in_file", "replace_between_anchors",
	"execute_command", "list_files", "search_files",
	"list_background_processes", "check_background_process", "stop_background_process",
	"manage_todos", "web_search", "analyze_image",
	"mcp_search_tools", "mcp_list_tools", "mcp_call_tool",
	"retrieve_tool_result", "introspect", "attempt_completion",
}

// ParseToolCalls extracts all XML tool calls from the response text.
// Returns them in order of appearance.
func ParseToolCalls(text string) []ToolCall {
	type posToolCall struct {
		pos  int
		call ToolCall
	}
	var calls []posToolCall

	for _, toolName := range allToolNames {
		var pattern string
		if complexTools[toolName] {
			// Greedy: match to the LAST closing tag
			pattern = fmt.Sprintf(`(?s)<%s>(.*)</%s>`, toolName, toolName)
		} else {
			// Non-greedy: match to the FIRST closing tag
			pattern = fmt.Sprintf(`(?s)<%s>(.*?)</%s>`, toolName, toolName)
		}

		re := regexp.MustCompile(pattern)
		matches := re.FindAllStringSubmatchIndex(text, -1)
		for _, loc := range matches {
			fullMatch := text[loc[0]:loc[1]]
			inner := text[loc[2]:loc[3]]
			params := extractParams(inner)
			calls = append(calls, posToolCall{
				pos: loc[0],
				call: ToolCall{
					Name:   toolName,
					Params: params,
					Raw:    fullMatch,
				},
			})
		}
	}

	// Sort by position in original text
	for i := 1; i < len(calls); i++ {
		for j := i; j > 0 && calls[j].pos < calls[j-1].pos; j-- {
			calls[j], calls[j-1] = calls[j-1], calls[j]
		}
	}

	result := make([]ToolCall, len(calls))
	for i, pc := range calls {
		result[i] = pc.call
	}
	return result
}

// extractParams pulls <param>value</param> pairs from inner XML content.
func extractParams(inner string) map[string]string {
	params := map[string]string{}
	// Match opening tags, then find corresponding close tag
	openRe := regexp.MustCompile(`<(\w+)>`)
	matches := openRe.FindAllStringSubmatchIndex(inner, -1)
	for _, loc := range matches {
		name := inner[loc[2]:loc[3]]
		afterOpen := loc[1]
		closeTag := "</" + name + ">"
		closeIdx := strings.Index(inner[afterOpen:], closeTag)
		if closeIdx < 0 {
			continue
		}
		value := inner[afterOpen : afterOpen+closeIdx]
		// Don't overwrite — first occurrence wins
		if _, exists := params[name]; !exists {
			params[name] = value
		}
	}
	return params
}

// StripThinkingTags removes <think>/<thinking> blocks from text.
func StripThinkingTags(text string) string {
	// Closed tags
	re1 := regexp.MustCompile(`(?s)<think>.*?</think>`)
	re2 := regexp.MustCompile(`(?s)<thinking>.*?</thinking>`)
	text = re1.ReplaceAllString(text, "")
	text = re2.ReplaceAllString(text, "")
	// Unclosed tags (e.g., truncated output)
	re3 := regexp.MustCompile(`(?s)<think>.*$`)
	re4 := regexp.MustCompile(`(?s)<thinking>.*$`)
	text = re3.ReplaceAllString(text, "")
	text = re4.ReplaceAllString(text, "")
	return text
}

// StripToolTags removes all tool XML from text for clean display.
func StripToolTags(text string) string {
	for _, name := range allToolNames {
		var pattern string
		if complexTools[name] {
			pattern = fmt.Sprintf(`(?s)<%s>.*</%s>`, name, name)
		} else {
			pattern = fmt.Sprintf(`(?s)<%s>.*?</%s>`, name, name)
		}
		re := regexp.MustCompile(pattern)
		text = re.ReplaceAllString(text, "")
	}
	return strings.TrimSpace(text)
}
