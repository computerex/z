package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const autoPrompt = `[AUTONOMOUS CHECK] Review all agent status in ~/COS/agents/. For each:
- If status=error: read output.log and stderr.log to diagnose, then restart with fix
- If status=done: summarize result briefly, note if follow-up action needed (push, PR, etc)
- If status=running but pid is dead: mark as error, diagnose from logs
- If running and pid alive: note progress

Then check ~/COS/projects.json for any projects that can be advanced.

IMPORTANT: If everything is healthy and running with no action needed, respond with just "All clear." and nothing else. Only take action or give detail when something needs attention.`

// sendMessageToCOS pipes a message to the z harness (COS brain session).
func sendMessageToCOS(message string) {
	zExe := findZ()
	args := strings.Fields(zExe)
	args = append(args, "--session", "cos-brain")

	cfg := loadConfig()
	if cfg.Model != "" {
		args = append(args, "--model", cfg.Model)
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdin = strings.NewReader(message)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = cosDir

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "\033[90m[COS brain error: %v]\033[0m\n", err)
	}
}

// checkNotifications drains ~/COS/notify/ and returns finished agent IDs.
func checkNotifications() []string {
	entries, err := os.ReadDir(notifyDir)
	if err != nil {
		return nil
	}
	var finished []string
	for _, e := range entries {
		if !e.IsDir() {
			finished = append(finished, e.Name())
			os.Remove(filepath.Join(notifyDir, e.Name()))
		}
	}
	return finished
}

// runLoop is the main COS interactive loop.
func runLoop(initialMessage string) {
	ensureDirs()
	cfg := loadConfig()
	tickSec := cfg.TickInterval
	if tickSec <= 0 {
		tickSec = 3600
	}

	fmt.Println()
	fmt.Println("━━━ COS ━━━")
	fmt.Printf("\033[90mAlways-on mode. Type to steer. Ctrl+C to exit. Auto-check every %dm.\033[0m\n", tickSec/60)
	fmt.Println()

	if initialMessage != "" {
		fmt.Printf("\033[90m>\033[0m %s\n", initialMessage)
		sendMessageToCOS(initialMessage)
	}

	inputCh := make(chan string)
	go readInputLoop(inputCh)

	ticker := time.NewTicker(time.Duration(tickSec) * time.Second)
	defer ticker.Stop()

	for {
		// Check notifications
		if finished := checkNotifications(); len(finished) > 0 {
			agentsStr := strings.Join(finished, " ")
			fmt.Printf("\n\033[90m[agent finished: %s]\033[0m\n", agentsStr)
			sendMessageToCOS(fmt.Sprintf(
				"[AGENT COMPLETED] The following agents just finished: %s. "+
					"Check their output logs, summarize results, and take next steps "+
					"(push branch, create PR, spawn follow-up agent, etc). "+
					"Update projects.json accordingly.", agentsStr))
		}

		select {
		case line, ok := <-inputCh:
			if !ok {
				// stdin closed
				fmt.Println("\n\033[90mCOS exiting. Agents keep running.\033[0m")
				return
			}
			line = strings.TrimSpace(line)
			if line != "" {
				sendMessageToCOS(line)
				ticker.Reset(time.Duration(tickSec) * time.Second)
			}

		case <-ticker.C:
			fmt.Println("\033[90m[auto-check]\033[0m")
			sendMessageToCOS(autoPrompt)
		}
	}
}

// readInputLoop reads lines from stdin and sends them to ch.
// Exits (closes channel) on EOF or terminal error.
func readInputLoop(ch chan<- string) {
	defer close(ch)
	scanner := bufio.NewScanner(os.Stdin)
	// Increase buffer for long pastes
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for {
		fmt.Print("\033[90mcos>\033[0m ")
		if !scanner.Scan() {
			return
		}
		ch <- scanner.Text()
	}
}
