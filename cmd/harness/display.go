package main

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/charmbracelet/glamour"
	"golang.org/x/term"
)

// Display handles streaming output formatting with XML tag suppression.
type Display struct {
	mu          sync.Mutex
	status      *StatusLine
	isTTY       bool

	// XML stream filter state machine
	sfBuf        []byte   // Buffer for potential XML tag
	sfInTag      bool     // Currently inside a < > tag
	sfSuppress   bool     // Currently suppressing content inside a tool tag
	sfTagName    string   // Current tag name being built
	sfTagDone    bool     // Tag name complete (hit space/slash), stop appending
	sfDepth      int      // Nesting depth of suppressed tags
	sfClosing    bool     // Building a closing tag

	// Thinking state
	inThinking       bool
	thinkBuf         strings.Builder
	thinkShown       bool
	nativeReasoning  bool  // True when OnReasoning has been called (native reasoning active)

	// Reasoning stream state
	reasoningLineStart bool
	reasoningHasText   bool
}

func NewDisplay(status *StatusLine) *Display {
	return &Display{
		status: status,
		isTTY:  term.IsTerminal(int(os.Stdout.Fd())),
	}
}

// toolTagNames are XML tags that should be suppressed from display.
var toolTagNames = map[string]bool{
	"read_file": true, "write_to_file": true, "replace_in_file": true,
	"replace_between_anchors": true, "execute_command": true,
	"list_files": true, "search_files": true, "manage_todos": true,
	"web_search": true, "list_background_processes": true,
	"check_background_process": true, "stop_background_process": true,
	"analyze_image": true, "mcp_search_tools": true, "mcp_list_tools": true,
	"mcp_call_tool": true, "retrieve_tool_result": true,
	"introspect": true, "attempt_completion": true,
}

// OnChunk processes a streaming chunk through the XML filter.
// Returns the displayable text (tool XML suppressed).
func (d *Display) OnChunk(chunk string) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	var output strings.Builder

	for _, ch := range chunk {
		result := d.sfChar(ch)
		if result != "" {
			output.WriteString(result)
		}
	}

	return output.String()
}

// OnReasoning displays reasoning/thinking content.
func (d *Display) OnReasoning(chunk string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.nativeReasoning = true

	if !d.thinkShown {
		d.thinkShown = true
		if d.status != nil {
			d.status.Clear()
		}
		if d.isTTY {
			fmt.Print("\n\033[2m\033[3mThinking:\033[0m\n")
		} else {
			fmt.Print("\nThinking:\n")
		}
	}

	// Stream reasoning char by char with dim styling to stdout
	for _, c := range chunk {
		if d.reasoningLineStart {
			// Skip leading blank lines
			if c == '\n' && !d.reasoningHasText {
				continue
			}
			fmt.Print("  ")
			d.reasoningLineStart = false
		}
		if d.isTTY && c != '\n' {
			fmt.Printf("\033[2m%c\033[0m", c)
		} else {
			fmt.Printf("%c", c)
		}
		if c != '\n' && c != '\r' && c != '\t' && c != ' ' {
			d.reasoningHasText = true
		}
		if c == '\n' {
			d.reasoningLineStart = true
			d.reasoningHasText = false
		}
	}
}

// FlushThinking outputs any buffered thinking content and ensures a clean
// line break after reasoning/thinking output.
func (d *Display) FlushThinking() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.inThinking && d.thinkBuf.Len() > 0 {
		d.printDim(d.thinkBuf.String())
		d.thinkBuf.Reset()
	}

	// Ensure cursor is on a fresh line after reasoning/thinking output
	if d.thinkShown && !d.reasoningLineStart {
		fmt.Println()
	}

	d.thinkShown = false
	d.inThinking = false
}

// RenderFinalContent renders the complete assistant response after streaming.
func (d *Display) RenderFinalContent(content string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Strip thinking tags and tool tags for display
	display := StripThinkingTags(content)
	display = StripToolTags(display)
	display = strings.TrimSpace(display)

	if display == "" {
		return
	}

	if d.isTTY {
		d.printFinal(display)
	} else {
		fmt.Println(display)
	}
}

// Reset clears the display state for a new response.
func (d *Display) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.sfBuf = nil
	d.sfInTag = false
	d.sfSuppress = false
	d.sfTagName = ""
	d.sfTagDone = false
	d.sfDepth = 0
	d.sfClosing = false
	d.inThinking = false
	d.thinkBuf.Reset()
	d.thinkShown = false
	d.nativeReasoning = false
	d.reasoningLineStart = true
	d.reasoningHasText = false
}

// sfChar is the char-by-char XML stream filter state machine.
// Returns empty string for suppressed chars, the char otherwise.
func (d *Display) sfChar(ch rune) string {
	switch {
	case ch == '<' && !d.sfSuppress:
		// Start of a potential tag (only when not suppressing)
		d.sfBuf = []byte{'<'}
		d.sfInTag = true
		d.sfClosing = false
		d.sfTagName = ""
		d.sfTagDone = false
		return ""

	case d.sfInTag:
		d.sfBuf = append(d.sfBuf, []byte(string(ch))...)

		if ch == '/' && len(d.sfBuf) == 2 {
			// Closing tag: </
			d.sfClosing = true
			return ""
		}

		if ch == '>' {
			// Tag complete
			d.sfInTag = false
			tagName := d.sfTagName

			if d.sfClosing {
				// Closing tag
				if toolTagNames[tagName] {
					if d.sfDepth > 0 {
						d.sfDepth--
					}
					if d.sfDepth == 0 {
						d.sfSuppress = false
					}
					d.sfBuf = nil
					return ""
				}
				if tagName == "thinking" || tagName == "think" {
					if d.nativeReasoning {
						// Native reasoning active — suppress thinking tags silently
						d.inThinking = false
						d.sfBuf = nil
						return ""
					}
					d.inThinking = false
					if d.thinkBuf.Len() > 0 {
						d.printDim(d.thinkBuf.String())
						d.thinkBuf.Reset()
					}
					d.sfBuf = nil
					return ""
				}
			} else {
				// Opening tag
				if toolTagNames[tagName] {
					d.sfSuppress = true
					d.sfDepth++
					d.sfBuf = nil
					return ""
				}
				if tagName == "thinking" || tagName == "think" {
					if d.nativeReasoning {
						// Native reasoning active — suppress thinking tags silently
						// Set inThinking to eat the content between tags
						d.inThinking = true
						d.sfBuf = nil
						return ""
					}
					d.inThinking = true
					if !d.thinkShown {
						d.thinkShown = true
						if d.isTTY {
							fmt.Print("\n\033[2m\033[3mThinking:\033[0m\n")
						} else {
							fmt.Print("\nThinking:\n")
						}
					}
					d.sfBuf = nil
					return ""
				}
			}

			// Not a tool/thinking tag — flush the buffered tag
			result := string(d.sfBuf)
			d.sfBuf = nil
			if d.sfSuppress {
				return ""
			}
			return result
		}

		// Building tag name — stop at first space, slash (after pos 1), or special char
		if !d.sfTagDone && ch != ' ' && ch != '\n' && ch != '\t' {
			if ch == '/' && len(d.sfBuf) > 2 {
				// Self-closing slash (not the leading slash of </tag>)
				d.sfTagDone = true
			} else if ch != '/' {
				d.sfTagName += string(ch)
			}
		} else if ch == ' ' || ch == '\n' || ch == '\t' {
			d.sfTagDone = true
		}
		return ""

	case d.sfSuppress:
		// Inside a suppressed tool tag — eat everything
		// But watch for the closing tag
		if ch == '<' {
			d.sfBuf = []byte{'<'}
			d.sfInTag = true
			d.sfClosing = false
			d.sfTagName = ""
			d.sfTagDone = false
		}
		return ""

	case d.inThinking:
		if d.nativeReasoning {
			// Native reasoning active — silently eat content inside <thinking> tags
			return ""
		}
		// Inside thinking tags — buffer for dim display
		d.thinkBuf.WriteRune(ch)
		if ch == '\n' && d.thinkBuf.Len() > 0 {
			line := strings.TrimRight(d.thinkBuf.String(), "\n")
			if line != "" {
				d.printDim("  " + line)
			}
			d.thinkBuf.Reset()
		}
		return ""

	default:
		return string(ch)
	}
}

// printDim outputs text in dim style to stdout.
func (d *Display) printDim(text string) {
	if d.status != nil {
		d.status.PrintSafe(func() {
			if d.isTTY {
				fmt.Printf("\033[2m%s\033[0m\n", text)
			} else {
				fmt.Printf("%s\n", text)
			}
		})
	} else {
		if d.isTTY {
			fmt.Printf("\033[2m%s\033[0m\n", text)
		} else {
			fmt.Printf("%s\n", text)
		}
	}
}

// IsTTY returns whether stdout is a terminal.
func (d *Display) IsTTY() bool {
	return d.isTTY
}

// renderMarkdown renders markdown text to styled terminal output using glamour.
func (d *Display) renderMarkdown(text string) string {
	width := 100
	if w, _, err := term.GetSize(int(os.Stdout.Fd())); err == nil && w > 20 {
		width = w
	}
	// Leave a 2-char margin on each side
	if width > 4 {
		width -= 4
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		return text
	}
	rendered, err := r.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimRight(rendered, "\n")
}

// printFinal outputs the final rendered content with markdown formatting.
func (d *Display) printFinal(text string) {
	rendered := d.renderMarkdown(text)
	if d.status != nil {
		d.status.PrintSafe(func() {
			fmt.Println()
			fmt.Println(rendered)
		})
	} else {
		fmt.Println()
		fmt.Println(rendered)
	}
}
