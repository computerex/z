package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
)

// Status phases
const (
	PhaseIdle       = "idle"
	PhaseSending    = "sending"
	PhaseStreaming   = "streaming"
	PhaseToolExec   = "tool_exec"
	PhaseRetrying   = "retrying"
	PhaseCompacting = "compacting"
	PhaseWaiting    = "waiting"
)

// Braille spinner frames
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// StatusLine renders a persistent single-line status at the bottom of the terminal.
type StatusLine struct {
	mu           sync.Mutex
	text         string
	phase        string
	iteration    int
	maxIter      int
	turnStart    time.Time
	phaseStart   time.Time
	spinIdx      int
	isTTY        bool
	lastRendered string
	ticker       *time.Ticker
	stopChan     chan struct{}
	running      bool
	safeMode     bool
	tokens       int
	maxTokens    int
}

func NewStatusLine() *StatusLine {
	sl := &StatusLine{
		phase:     PhaseIdle,
		isTTY:     term.IsTerminal(int(os.Stdout.Fd())),
		turnStart: time.Now(),
		stopChan:  make(chan struct{}),
		safeMode:  os.Getenv("HARNESS_SAFE_MODE") == "1",
	}
	return sl
}

// Update sets the status text and phase.
func (sl *StatusLine) Update(text, phase string) {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	if phase != sl.phase {
		sl.phaseStart = time.Now()
	}
	sl.text = text
	sl.phase = phase
	sl.render()

	if !sl.running {
		sl.running = true
		sl.ticker = time.NewTicker(120 * time.Millisecond)
		go sl.tickLoop()
	}
}

// SetIteration sets the iteration counter for display.
func (sl *StatusLine) SetIteration(iter, max int) {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.iteration = iter
	sl.maxIter = max
}

// SetTokens sets the context token usage for display.
func (sl *StatusLine) SetTokens(tokens, maxTokens int) {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.tokens = tokens
	sl.maxTokens = maxTokens
}

// NewTurn resets turn timer.
func (sl *StatusLine) NewTurn() {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.turnStart = time.Now()
}

// Clear removes the status line.
func (sl *StatusLine) Clear() {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	if sl.running {
		sl.running = false
		sl.ticker.Stop()
		close(sl.stopChan)
		sl.stopChan = make(chan struct{})
	}

	sl.phase = PhaseIdle
	if sl.isTTY && sl.lastRendered != "" {
		fmt.Fprintf(os.Stderr, "\r\033[2K")
		sl.lastRendered = ""
	}
}

// PrintSafe clears the status line, executes fn, then re-renders.
func (sl *StatusLine) PrintSafe(fn func()) {
	sl.mu.Lock()
	if sl.isTTY && sl.lastRendered != "" {
		fmt.Fprintf(os.Stderr, "\r\033[2K")
	}
	sl.mu.Unlock()

	fn()

	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.render()
}

func (sl *StatusLine) tickLoop() {
	for {
		select {
		case <-sl.ticker.C:
			sl.mu.Lock()
			sl.spinIdx = (sl.spinIdx + 1) % len(spinnerFrames)
			sl.render()
			sl.mu.Unlock()
		case <-sl.stopChan:
			return
		}
	}
}

func (sl *StatusLine) render() {
	if !sl.isTTY {
		return
	}

	width, _, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil || width < 20 {
		width = 80
	}

	icon := sl.phaseIcon()
	phaseElapsed := time.Since(sl.phaseStart).Round(time.Second)
	turnElapsed := time.Since(sl.turnStart).Round(time.Second)

	var parts []string

	if sl.safeMode {
		parts = append(parts, "[SAFE]")
	}

	parts = append(parts, icon)
	parts = append(parts, sl.text)

	if sl.iteration > 0 {
		parts = append(parts, fmt.Sprintf("│ iter: %d/%d", sl.iteration, sl.maxIter))
	}

	if sl.tokens > 0 && sl.maxTokens > 0 {
		pct := sl.tokens * 100 / sl.maxTokens
		parts = append(parts, fmt.Sprintf("│ ctx: %dk/%dk (%d%%)", sl.tokens/1000, sl.maxTokens/1000, pct))
	}

	parts = append(parts, fmt.Sprintf("│ %s", phaseElapsed))

	if sl.iteration > 0 {
		parts = append(parts, fmt.Sprintf("│ turn %s", turnElapsed))
	}

	line := strings.Join(parts, " ")

	// Truncate to terminal width
	if len(line) > width-2 {
		line = line[:width-5] + "..."
	}

	// Dim the entire line
	rendered := fmt.Sprintf("\r\033[2K\033[2m%s\033[0m", line)
	sl.lastRendered = line
	fmt.Fprint(os.Stderr, rendered)
}

func (sl *StatusLine) phaseIcon() string {
	accent := "\033[38;5;75m" // Blue
	reset := "\033[0m"
	switch sl.phase {
	case PhaseIdle:
		return accent + "○" + reset
	case PhaseStreaming:
		return accent + "●" + reset
	case PhaseSending, PhaseToolExec, PhaseCompacting, PhaseWaiting, PhaseRetrying:
		return accent + spinnerFrames[sl.spinIdx] + reset
	default:
		return accent + "○" + reset
	}
}
