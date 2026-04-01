//go:build !windows

package main

import (
	"os"
	"syscall"
)

func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func killPID(pid int) error {
	if pid <= 0 {
		return nil
	}
	// Kill the process group (negative PID) for tree kill
	pgid, err := syscall.Getpgid(pid)
	if err == nil && pgid > 0 {
		syscall.Kill(-pgid, syscall.SIGKILL)
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return proc.Kill()
}
