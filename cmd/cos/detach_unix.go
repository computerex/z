//go:build !windows

package main

import (
	"os/exec"
	"syscall"
)

// detachProcess configures a command to run in a new session (setsid)
// so it survives the parent exiting.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}
}
