// cos — Chief of Staff CLI
//
// Always-on engineering orchestrator. Manages sub-agents via the z harness.
// Cross-platform: builds to a single binary on Windows, Linux, macOS.
//
// Usage:
//
//	cos                      Start interactive loop
//	cos "do something"       Start loop with initial message
//	cos --agents             List all agents
//	cos --log <id>           Show agent log
//	cos --kill <id>          Kill a running agent
//	cos --reset              Reset session
//
// Sub-agent management (used by COS brain):
//
//	cos agent spawn <id> --prompt "..." [--task "..."] [--worktree /path] [--model m]
//	cos agent list
//	cos agent log <id> [--tail N]
//	cos agent kill <id>
//	cos agent resume <id> --prompt "..."
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	initPaths()
	ensureDirs()

	args := os.Args[1:]

	// Route to agent subcommand if present
	if len(args) > 0 && args[0] == "agent" {
		runAgentSubcommand(args[1:])
		return
	}

	// Parse top-level flags
	switch {
	case hasFlag(args, "--help") || hasFlag(args, "-h"):
		printHelp()
	case hasFlag(args, "--agents"):
		cmdListAgents()
	case hasFlag(args, "--reset"):
		cmdReset()
	case flagVal(args, "--log") != "":
		id := flagVal(args, "--log")
		tail := flagIntVal(args, "--tail", 0)
		cmdShowLog(id, tail)
	case flagVal(args, "--kill") != "":
		cmdKillAgent(flagVal(args, "--kill"))
	default:
		// Everything else is an initial message
		msg := strings.Join(positionalArgs(args), " ")
		runLoop(msg)
	}
}

func cmdReset() {
	os.Remove(sessionFile)
	fmt.Println("Session reset.")
}

func printHelp() {
	fmt.Print(`cos — Chief of Staff

Interactive (persistent loop):
  cos                      Start COS loop (monitors + accepts input)
  cos "do something"       Start loop with an initial message

One-shot:
  cos --agents             List all agents (running + done)
  cos --log <id>           Show agent log
  cos --kill <id>          Kill a running agent
  cos --reset              Fresh COS session
  cos --help               This help

Sub-agent management (used by COS brain via execute_command):
  cos agent spawn <id> --prompt "..." [--task "..."] [--worktree /p] [--model m]
  cos agent list
  cos agent log <id> [--tail N]
  cos agent kill <id>
  cos agent resume <id> --prompt "..."

Inside the loop:
  Type anything             Send a message to COS
  Ctrl+C                    Exit (agents keep running)
`)
}

// runAgentSubcommand handles "cos agent <action> ..."
func runAgentSubcommand(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: cos agent <spawn|list|log|kill|resume> ...")
		return
	}
	action := args[0]
	rest := args[1:]

	switch action {
	case "_wait":
		// Internal reaper: cos agent _wait <id> <pid>
		if len(rest) < 2 {
			return
		}
		var pid int
		fmt.Sscanf(rest[1], "%d", &pid)
		if pid > 0 {
			waitForAgent(rest[0], pid)
		}
		return

	case "spawn":
		if len(rest) == 0 {
			fmt.Println("Usage: cos agent spawn <id> --prompt \"...\" [--task \"...\"] [--worktree /path] [--model m]")
			return
		}
		id := rest[0]
		prompt := flagVal(rest, "--prompt")
		task := flagVal(rest, "--task")
		worktree := flagVal(rest, "--worktree")
		model := flagVal(rest, "--model")
		if prompt == "" {
			fmt.Println("Error: --prompt is required")
			return
		}
		pid, err := spawnAgent(id, prompt, task, worktree, model)
		if err != nil {
			fmt.Printf("Error spawning agent: %v\n", err)
			return
		}
		fmt.Printf("Spawned agent '%s' (PID %d)\n", id, pid)

	case "list":
		cmdListAgents()

	case "log":
		if len(rest) == 0 {
			fmt.Println("Usage: cos agent log <id> [--tail N]")
			return
		}
		tail := flagIntVal(rest, "--tail", 0)
		cmdShowLog(rest[0], tail)

	case "kill":
		if len(rest) == 0 {
			fmt.Println("Usage: cos agent kill <id>")
			return
		}
		cmdKillAgent(rest[0])

	case "resume":
		if len(rest) == 0 {
			fmt.Println("Usage: cos agent resume <id> --prompt \"...\"")
			return
		}
		id := rest[0]
		prompt := flagVal(rest, "--prompt")
		if prompt == "" {
			fmt.Println("Error: --prompt is required")
			return
		}
		pid, err := resumeAgent(id, prompt)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}
		fmt.Printf("Resumed agent '%s' with follow-up (PID %d)\n", id, pid)

	default:
		fmt.Printf("Unknown action: %s\n", action)
		fmt.Println("Usage: cos agent <spawn|list|log|kill|resume> ...")
	}
}

// --- Minimal flag parsing helpers (no external deps) ---

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func flagVal(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func flagIntVal(args []string, flag string, def int) int {
	s := flagVal(args, flag)
	if s == "" {
		return def
	}
	var v int
	fmt.Sscanf(s, "%d", &v)
	return v
}

func positionalArgs(args []string) []string {
	var result []string
	skip := false
	flagsWithVal := map[string]bool{"--log": true, "--kill": true, "--tail": true, "--model": true}
	for _, a := range args {
		if skip {
			skip = false
			continue
		}
		if strings.HasPrefix(a, "--") {
			if flagsWithVal[a] {
				skip = true
			}
			continue
		}
		result = append(result, a)
	}
	return result
}
