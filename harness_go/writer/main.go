package main

import (
        "os"
)

func main() {
        content := "package main\n\nfunc getSystemPrompt(workspace string) string {\n        return \"test\"\n}\n"
        os.WriteFile("prompts.go", []byte(content), 0644)
}