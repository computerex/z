package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

// Version info
const Version = "1.0.0"

func main() {
	exePath, err := os.Executable()
	if err != nil {
		exePath = "."
	}
	harnessDir := filepath.Dir(exePath)

	args := os.Args[1:]

	if len(args) > 0 && (args[0] == "--install" || args[0] == "-i") {
		handleInstall(args[1:])
		return
	}

	if len(args) > 0 && (args[0] == "--version" || args[0] == "-v") {
		fmt.Printf("Harness v%s\n", Version)
		return
	}

	if len(args) > 0 && (args[0] == "--help" || args[0] == "-h") {
		printHelp()
		return
	}

	workspace := "."
	if len(args) > 0 && !isFlag(args[0]) {
		workspace = args[0]
		args = args[1:]
	}

	if err := os.Chdir(workspace); err != nil {
		fmt.Fprintf(os.Stderr, "Error: Cannot access workspace %s: %v\n", workspace, err)
		os.Exit(1)
	}

	workspace, _ = filepath.Abs(workspace)

	config, err := FindConfig(workspace)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Run './harness --install' to set up configuration.\n")
		os.Exit(1)
	}

	if err := config.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if err := runApp(harnessDir, workspace, config, args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runApp(harnessDir, workspace string, config *Config, args []string) error {
	// Check if stdin has data (piped input)
	stat, _ := os.Stdin.Stat()
	hasPipedInput := (stat.Mode() & os.ModeCharDevice) == 0

	if hasPipedInput {
		// Piped input - single request mode
		data := make([]byte, 0, 1024)
		buf := make([]byte, 4096)
		for {
			n, err := os.Stdin.Read(buf)
			if n > 0 {
				data = append(data, buf[:n]...)
			}
			if err != nil {
				break
			}
		}

		input := string(data)
		if input == "" {
			fmt.Fprintln(os.Stderr, "No input provided.")
			os.Exit(1)
		}

		app, err := NewApp(workspace, config)
		if err != nil {
			return fmt.Errorf("failed to create app: %w", err)
		}

		defer CleanupBackgroundProcs()
		return app.RunSingle(input)
	}

	// Interactive mode
	app, err := NewApp(workspace, config)
	if err != nil {
		return fmt.Errorf("failed to create app: %w", err)
	}

	defer CleanupBackgroundProcs()
	return app.Run()
}

func printHelp() {
	fmt.Printf(`Harness v%s - AI Coding Assistant

Usage: harness [workspace] [options]

Arguments:
  workspace         Workspace directory (default: current directory)

Options:
  -i, --install    Run setup wizard
  -v, --version    Show version
  -h, --help       Show this help message

`, Version)
}

func isFlag(arg string) bool {
	return len(arg) > 0 && (arg[0] == '-' || arg[0] == '/')
}

// Render markup to plain text (strips tags like [dim], [bold], etc.)
func Render(text string) string {
	// Define markup patterns
	patterns := []string{
		`\[dim\]`,          // dim
		`\[/dim\]`,         // /dim
		`\[bold\]`,         // bold
		`\[/bold\]`,        // /bold
		`\[red\]`,          // red
		`\[/red\]`,         // /red
		`\[green\]`,        // green
		`\[/green\]`,       // /green
		`\[cyan\]`,         // cyan
		`\[/cyan\]`,        // /cyan
		`\[blue\]`,         // blue
		`\[/blue\]`,        // /blue
		`\[yellow\]`,       // yellow
		`\[/yellow\]`,      // /yellow
		`\[white\]`,        // white
		`\[/white\]`,       // /white
		`\[italic\]`,       // italic
		`\[/italic\]`,      // /italic
		`\[underline\]`,   // underline
		`\[/underline\]`,  // /underline
		`\[strike\]`,       // strike
		`\[/strike\]`,      // /strike
	}

	for _, pattern := range patterns {
		text = regexp.MustCompile(pattern).ReplaceAllString(text, "")
	}

	return text
}