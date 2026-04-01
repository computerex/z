//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// detachProcess configures a command to run as a detached process on Windows.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}
