//go:build windows

package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	cmd := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH", "/FO", "CSV")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), strconv.Itoa(pid))
}

func killPID(pid int) error {
	if pid <= 0 {
		return nil
	}
	// taskkill /F /T /PID kills the entire tree
	return exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(pid)).Run()
}
