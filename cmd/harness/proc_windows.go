package main

import (
	"fmt"
	"os/exec"
	"syscall"
)

// setProcGroup creates a new process group on Windows using CREATE_NEW_PROCESS_GROUP.
func setProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// killProcessTree terminates a process tree on Windows using taskkill.
func killProcessTree(pid int) error {
	kill := exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprintf("%d", pid))
	return kill.Run()
}
