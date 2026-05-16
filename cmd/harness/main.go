// Package main implements an agentic coding harness in Go.
// This is a 1:1 behavioral port of the Python harness.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/term"
)

var version = "0.1.0"

func main() {
	workspace := flag.String("workspace", ".", "Workspace directory")
	newSess := flag.Bool("new", false, "Start fresh session")
	sessionName := flag.String("session", "default", "Session name")
	listSess := flag.Bool("list", false, "List all sessions")
	install := flag.Bool("install", false, "Run setup wizard")
	debug := flag.Bool("debug", false, "Enable debug logging")
	showHelp := flag.Bool("help", false, "Show help")
	flag.Parse()

	if *showHelp {
		printUsage()
		return
	}

	if *install {
		runInstallWizard()
		return
	}

	// Resolve workspace
	ws := *workspace
	if ws == "." {
		var err error
		ws, err = os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting cwd: %v\n", err)
			os.Exit(1)
		}
	} else {
		var err error
		ws, err = filepath.Abs(ws)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error resolving workspace: %v\n", err)
			os.Exit(1)
		}
	}

	if err := os.Chdir(ws); err != nil {
		fmt.Fprintf(os.Stderr, "Cannot chdir to %s: %v\n", ws, err)
		os.Exit(1)
	}

	// Init logging
	initLogging(ws, *sessionName)
	if *debug {
		enableDebug()
	}
	logInfo("=== Harness (Go) starting === workspace=%s session=%s new=%v", ws, *sessionName, *newSess)

	// Load config from ~/.z.json
	cfg := loadConfig()
	logInfo("Config loaded: provider=%s model=%s api_base=%s", cfg.Provider, cfg.Model, cfg.APIBase)

	isTTY := term.IsTerminal(int(os.Stdout.Fd()))

	if *listSess {
		sessions := listAllSessions(ws)
		if len(sessions) == 0 {
			fmt.Println("No saved sessions.")
		} else {
			for _, s := range sessions {
				fmt.Println("  " + s)
			}
		}
		return
	}

	// Create agent
	sessionPath := getSessionPath(ws, *sessionName)
	agent := NewAgent(ws, cfg, sessionPath)

	// Load or create session
	if !*newSess {
		if _, err := os.Stat(sessionPath); err == nil {
			sess, loadErr := LoadSession(sessionPath)
			if loadErr == nil {
				agent.SetSession(sess)
				sess.Name = *sessionName
				msgCount := len(sess.Messages)
				if msgCount > 1 {
					msgCount-- // Don't count system prompt
				}
				logInfo("Resumed session '%s' (%d messages)", *sessionName, msgCount)
				if isTTY {
					fmt.Printf("  \033[2mResumed \033[37m%s\033[0m\033[2m (%d messages)\033[0m\n", *sessionName, msgCount)
				} else {
					fmt.Printf("  Resumed %s (%d messages)\n", *sessionName, msgCount)
				}
			} else {
				logWarn("Failed to load session: %v", loadErr)
			}
		}
	} else {
		logInfo("New session '%s'", *sessionName)
		if isTTY {
			fmt.Printf("  \033[2mNew session \033[37m%s\033[0m\n", *sessionName)
		} else {
			fmt.Printf("  New session %s\n", *sessionName)
		}
	}

	// Print banner
	printBanner(cfg, ws, *sessionName, isTTY)

	// Check if stdin is piped
	stat, _ := os.Stdin.Stat()
	if (stat.Mode() & os.ModeCharDevice) == 0 {
		RunPiped(agent)
		return
	}

	// Interactive REPL
	repl := NewREPL(ws, cfg, agent, sessionPath)
	if err := repl.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "REPL error: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`harness v%s — agentic coding assistant (Go)

Usage: harness [options] [workspace]

Options:
  --workspace DIR   Workspace directory (default: .)
  --new             Start fresh session
  --session NAME    Session name (default: "default")
  --list            List all sessions
  --install         Run setup wizard
  --debug           Enable debug logging
  --help            Show this help
`, version)
}

func printBanner(cfg *Config, workspace, sessionName string, isTTY bool) {
	fmt.Println()
	if isTTY {
		fmt.Printf("  \033[1;36mharness\033[0m \033[2mv%s (Go)\033[0m\n", version)
		fmt.Printf("  \033[2m%s • %s\033[0m\n", cfg.Provider, cfg.Model)
		fmt.Printf("  \033[2m%s\033[0m\n", workspace)
		fmt.Println()
		fmt.Println("  \033[2mType /help for commands, Ctrl+C twice to interrupt\033[0m")
	} else {
		fmt.Printf("  harness v%s (Go)\n", version)
		fmt.Printf("  %s • %s\n", cfg.Provider, cfg.Model)
		fmt.Printf("  %s\n", workspace)
		fmt.Println()
		fmt.Println("  Type /help for commands, Ctrl+C twice to interrupt")
	}
	fmt.Println()
}
