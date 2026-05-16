package main

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"time"
)

// InterruptState tracks whether the user has requested an interrupt.
type InterruptState struct {
	mu          sync.Mutex
	interrupted bool
	reason      string
}

var globalInterrupt = &InterruptState{}

func (is *InterruptState) Reset() {
	is.mu.Lock()
	defer is.mu.Unlock()
	is.interrupted = false
	is.reason = ""
}

func (is *InterruptState) Trigger(reason string) {
	is.mu.Lock()
	defer is.mu.Unlock()
	is.interrupted = true
	is.reason = reason
	logInfo("INTERRUPT triggered: reason=%s", reason)
}

func (is *InterruptState) IsInterrupted() bool {
	is.mu.Lock()
	defer is.mu.Unlock()
	return is.interrupted
}

func (is *InterruptState) Snapshot() (bool, string) {
	is.mu.Lock()
	defer is.mu.Unlock()
	return is.interrupted, is.reason
}

// KeyboardMonitor handles Ctrl+C with double-tap detection.
type KeyboardMonitor struct {
	mu                sync.Mutex
	running           bool
	sigChan           chan os.Signal
	lastSigintTime    time.Time
	doubleTapWindow   time.Duration
	alreadyInterrupted bool
}

func NewKeyboardMonitor() *KeyboardMonitor {
	return &KeyboardMonitor{
		doubleTapWindow: 1500 * time.Millisecond,
	}
}

func (km *KeyboardMonitor) Start() {
	km.mu.Lock()
	defer km.mu.Unlock()

	// Always reset interrupt state to clear stale flags from previous turns
	globalInterrupt.Reset()
	km.lastSigintTime = time.Time{}
	km.alreadyInterrupted = false

	if km.running {
		return
	}
	km.running = true

	km.sigChan = make(chan os.Signal, 2)
	signal.Notify(km.sigChan, os.Interrupt)

	go km.monitorLoop()
}

func (km *KeyboardMonitor) Stop() {
	km.mu.Lock()
	defer km.mu.Unlock()

	if !km.running {
		return
	}
	km.running = false

	signal.Stop(km.sigChan)
	close(km.sigChan)
	km.sigChan = nil
}

func (km *KeyboardMonitor) monitorLoop() {
	for sig := range km.sigChan {
		if sig == nil {
			return
		}
		km.handleSigint()
	}
}

func (km *KeyboardMonitor) handleSigint() {
	now := time.Now()

	// Already interrupted → hard exit
	if globalInterrupt.IsInterrupted() {
		logWarn("Ctrl+C while already interrupted — hard exit")
		fmt.Fprintln(os.Stderr, "\n  Force exit")
		os.Exit(130)
	}

	km.mu.Lock()
	elapsed := now.Sub(km.lastSigintTime)
	km.lastSigintTime = now
	km.mu.Unlock()

	if elapsed <= km.doubleTapWindow && elapsed > 0 {
		// Double-tap confirmed → soft interrupt
		logInfo("Double-tap Ctrl+C confirmed (%.2fs apart) — interrupting", elapsed.Seconds())
		globalInterrupt.Trigger("ctrl-c")
	} else {
		// First tap — print hint
		logInfo("Single Ctrl+C (%.2fs since last) — waiting for double-tap", elapsed.Seconds())
		fmt.Fprintln(os.Stderr, "\n  Press Ctrl+C again to interrupt")
	}
}

// Convenience functions

func isInterrupted() bool {
	return globalInterrupt.IsInterrupted()
}

func resetInterrupt() {
	globalInterrupt.Reset()
}
