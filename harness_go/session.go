package main

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Session represents a chat session
type Session struct {
	Workspace    string            `json:"workspace"`
	Messages     []Message         `json:"messages"`
	ContextItems []ContextItem     `json:"context_items,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	ModifiedAt  time.Time         `json:"modified_at"`
}

// SessionManager manages chat sessions
type SessionManager struct {
	sessionsDir string
}

// NewSessionManager creates a new session manager
func NewSessionManager(sessionsDir string) (*SessionManager, error) {
	if err := os.MkdirAll(sessionsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create sessions directory: %w", err)
	}
	return &SessionManager{sessionsDir: sessionsDir}, nil
}

// getWorkspaceHash creates a unique hash for a workspace directory
func (sm *SessionManager) getWorkspaceHash(workspace string) string {
	hash := md5.Sum([]byte(workspace))
	return fmt.Sprintf("%x", hash)[:12]
}

// getSessionPath returns the path for a session file
func (sm *SessionManager) getSessionPath(workspace string, sessionName string) string {
	workspaceHash := sm.getWorkspaceHash(workspace)
	workspaceDir := filepath.Join(sm.sessionsDir, workspaceHash)
	os.MkdirAll(workspaceDir, 0755)
	return filepath.Join(workspaceDir, sessionName+".json")
}

// Save saves a session to disk
func (sm *SessionManager) Save(workspace string, sessionName string, messages []Message, contextItems []ContextItem) error {
	path := sm.getSessionPath(workspace, sessionName)

	session := Session{
		Workspace:    workspace,
		Messages:     messages,
		ContextItems: contextItems,
		ModifiedAt:   time.Now(),
	}

	// Try to load existing to preserve created_at
	if existing, err := sm.Load(workspace, sessionName); err == nil {
		session.CreatedAt = existing.CreatedAt
	} else {
		session.CreatedAt = time.Now()
	}

	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write session file: %w", err)
	}

	return nil
}

// Load loads a session from disk
func (sm *SessionManager) Load(workspace string, sessionName string) (*Session, error) {
	path := sm.getSessionPath(workspace, sessionName)

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("session not found: %w", err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("failed to parse session: %w", err)
	}

	return &session, nil
}

// List lists all sessions for a workspace
func (sm *SessionManager) List(workspace string) ([]SessionInfo, error) {
	workspaceHash := sm.getWorkspaceHash(workspace)
	workspaceDir := filepath.Join(sm.sessionsDir, workspaceHash)

	entries, err := os.ReadDir(workspaceDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []SessionInfo{}, nil
		}
		return nil, fmt.Errorf("failed to read sessions directory: %w", err)
	}

	var sessions []SessionInfo
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		sessionName := entry.Name()[:len(entry.Name())-5]
		path := filepath.Join(workspaceDir, entry.Name())

		info, err := os.Stat(path)
		if err != nil {
			continue
		}

		msgCount := 0
		if session, err := sm.Load(workspace, sessionName); err == nil {
			msgCount = len(session.Messages) - 1 // minus system prompt
			if msgCount < 0 {
				msgCount = 0
			}
		}

		sessions = append(sessions, SessionInfo{
			Name:        sessionName,
			ModifiedAt:  info.ModTime(),
			MessageCount: msgCount,
		})
	}

	// Sort by most recently modified
	for i := range sessions {
		for j := i + 1; j < len(sessions); j++ {
			if sessions[j].ModifiedAt.After(sessions[i].ModifiedAt) {
				sessions[i], sessions[j] = sessions[j], sessions[i]
			}
		}
	}

	return sessions, nil
}

// SessionInfo holds basic session metadata
type SessionInfo struct {
	Name         string
	ModifiedAt   time.Time
	MessageCount int
}

// Delete deletes a session file
func (sm *SessionManager) Delete(workspace string, sessionName string) error {
	path := sm.getSessionPath(workspace, sessionName)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("session not found")
	}
	return os.Remove(path)
}

// GetSessionsDir returns the sessions directory for a workspace
func (sm *SessionManager) GetSessionsDir(workspace string) string {
	workspaceHash := sm.getWorkspaceHash(workspace)
	return filepath.Join(sm.sessionsDir, workspaceHash)
}