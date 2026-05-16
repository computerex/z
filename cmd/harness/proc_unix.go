// +build !windows

package main

import (
	"os/exec"
	"syscall"
)

// setProcGroup puts the command in its own process group on Unix.
func setProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}

// killProcessTree kills a process and all its children on Unix.
func killProcessTree(pid int) error {
	// Kill the process group (negative PID)
	return syscall.Kill(-pid, syscall.SIGKILL)
}
