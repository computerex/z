package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	logMu     sync.Mutex
	logFile   *os.File
	logDebug  bool
	logStdout = false // Set true for debug
)

func initLogging(workspace, session string) {
	dir := filepath.Join(workspace, ".harness_output")
	os.MkdirAll(dir, 0755)
	path := filepath.Join(dir, "harness.log")

	var err error
	logFile, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Printf("Warning: cannot open log file %s: %v", path, err)
	}
}

func enableDebug() {
	logDebug = true
}

func logMsg(level, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	ts := time.Now().Format("15:04:05.000")
	line := fmt.Sprintf("[%s] %s %s\n", ts, level, msg)

	logMu.Lock()
	defer logMu.Unlock()

	if logFile != nil {
		logFile.WriteString(line)
	}
	if logStdout {
		fmt.Fprint(os.Stderr, line)
	}
}

func logInfo(format string, args ...interface{})  { logMsg("INFO", format, args...) }
func logWarn(format string, args ...interface{})   { logMsg("WARN", format, args...) }
func logError(format string, args ...interface{})  { logMsg("ERROR", format, args...) }
func logDebugf(format string, args ...interface{}) {
	if logDebug {
		logMsg("DEBUG", format, args...)
	}
}
