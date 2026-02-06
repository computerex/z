package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// ContextItem represents an item in the context container
type ContextItem struct {
	ID        int
	Type      string // 'file', 'fragment', 'command_output', 'search_result'
	Source    string // path or command
	Content   string
	AddedAt   float64
	LineRange *[2]int // for file fragments
}

// Summary returns a short summary of this item
func (ci *ContextItem) Summary() string {
	lines := strings.Count(ci.Content, "\n") + 1
	size := len(ci.Content)
	age := int(time.Now().Unix()) - int(ci.AddedAt)
	ageStr := fmt.Sprintf("%ds", age)
	if age >= 60 {
		ageStr = fmt.Sprintf("%dm", age/60)
	}

	switch ci.Type {
	case "file":
		if ci.LineRange != nil {
			return fmt.Sprintf("[%d] file: %s (L%d-%d, %dL, %dB, %s ago)", ci.ID, ci.Source, (*ci.LineRange)[0], (*ci.LineRange)[1], lines, size, ageStr)
		}
		return fmt.Sprintf("[%d] file: %s (%dL, %dB, %s ago)", ci.ID, ci.Source, lines, size, ageStr)
	case "command_output":
		cmdShort := ci.Source
		if len(cmdShort) > 40 {
			cmdShort = cmdShort[:40] + "..."
		}
		return fmt.Sprintf("[%d] cmd: %s (%dL, %dB, %s ago)", ci.ID, cmdShort, lines, size, ageStr)
	case "search_result":
		return fmt.Sprintf("[%d] search: %s (%d matches, %s ago)", ci.ID, ci.Source, lines, ageStr)
	default:
		return fmt.Sprintf("[%d] %s: %s (%dL, %s ago)", ci.ID, ci.Type, ci.Source, lines, ageStr)
	}
}

// ContextContainer manages the agent's working context
type ContextContainer struct {
	items  map[int]ContextItem
	nextID int
}

// NewContextContainer creates a new context container
func NewContextContainer() *ContextContainer {
	return &ContextContainer{
		items:  make(map[int]ContextItem),
		nextID: 1,
	}
}

// Add adds an item to context
func (cc *ContextContainer) Add(itemType, source, content string, lineRange *[2]int) int {
	itemID := cc.nextID
	cc.nextID++
	cc.items[itemID] = ContextItem{
		ID:        itemID,
		Type:      itemType,
		Source:    source,
		Content:   content,
		AddedAt:   float64(time.Now().Unix()),
		LineRange: lineRange,
	}
	return itemID
}

// Remove removes an item from context
func (cc *ContextContainer) Remove(itemID int) bool {
	if _, ok := cc.items[itemID]; ok {
		delete(cc.items, itemID)
		return true
	}
	return false
}

// RemoveBySource removes all items with matching source
func (cc *ContextContainer) RemoveBySource(source string) int {
	var toRemove []int
	for id, item := range cc.items {
		if strings.Contains(item.Source, source) {
			toRemove = append(toRemove, id)
		}
	}
	for _, id := range toRemove {
		delete(cc.items, id)
	}
	return len(toRemove)
}

// Get returns an item by ID
func (cc *ContextContainer) Get(itemID int) *ContextItem {
	if item, ok := cc.items[itemID]; ok {
		return &item
	}
	return nil
}

// ListItems returns all items
func (cc *ContextContainer) ListItems() []ContextItem {
	items := make([]ContextItem, 0, len(cc.items))
	for _, item := range cc.items {
		items = append(items, item)
	}
	return items
}

// TotalSize returns total character count
func (cc *ContextContainer) TotalSize() int {
	total := 0
	for _, item := range cc.items {
		total += len(item.Content)
	}
	return total
}

// Summary returns a summary of all context items
func (cc *ContextContainer) Summary() string {
	if len(cc.items) == 0 {
		return "Context is empty."
	}

	lines := []string{fmt.Sprintf("Context (%d items, %d chars):", len(cc.items), cc.TotalSize())}
	for _, item := range cc.items {
		lines = append(lines, "  "+item.Summary())
	}
	return strings.Join(lines, "\n")
}

// Clear clears all context items
func (cc *ContextContainer) Clear() {
	cc.items = make(map[int]ContextItem)
	cc.nextID = 1
}

// Count returns the number of items
func (cc *ContextContainer) Count() int {
	return len(cc.items)
}

// GetContextItems returns a slice of ContextItem for serialization
func (cc *ContextContainer) GetContextItems() []ContextItem {
	items := make([]ContextItem, 0, len(cc.items))
	for _, item := range cc.items {
		items = append(items, item)
	}
	return items
}

// LoadContextItems loads context items from a slice
func (cc *ContextContainer) LoadContextItems(items []ContextItem) {
	for _, item := range items {
		cc.items[item.ID] = item
		if item.ID >= cc.nextID {
			cc.nextID = item.ID + 1
		}
	}
}

// Token estimation constants
const (
	avgTokensPerWord = 1.33
	avgCharsPerToken = 4
)

// EstimateTokens estimates token count for text
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	// Simple estimation: ~4 characters per token
	return int(math.Ceil(float64(len(text)) / avgCharsPerToken))
}

// EstimateMessagesTokens estimates token count for messages
func EstimateMessagesTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		// Add overhead for role markers
		overhead := 10 // "<role>...</role>\n"
		total += EstimateTokens(msg.Content) + overhead
	}
	return total
}

// ModelLimits defines token limits for different models
type ModelLimits struct {
	ContextLimit int
	MaxAllowed   int
}

// GetModelLimits returns token limits for a model
func GetModelLimits(model string) (contextLimit int, maxAllowed int) {
	// Define limits for known models
	limits := map[string]ModelLimits{
		"glm-4.7":      {ContextLimit: 128000, MaxAllowed: 128000},
		"glm-4.6v":     {ContextLimit: 128000, MaxAllowed: 128000},
		"glm-4":        {ContextLimit: 128000, MaxAllowed: 128000},
		"glm-4-plus":   {ContextLimit: 128000, MaxAllowed: 128000},
		"gpt-4":        {ContextLimit: 8192, MaxAllowed: 8192},
		"gpt-4-32k":    {ContextLimit: 32768, MaxAllowed: 32768},
		"gpt-4-turbo":  {ContextLimit: 128000, MaxAllowed: 128000},
		"claude-3-sonnet": {ContextLimit: 200000, MaxAllowed: 200000},
		"claude-3-opus":   {ContextLimit: 200000, MaxAllowed: 200000},
	}

	// Check for partial match
	for modelName, modelLimits := range limits {
		if strings.Contains(strings.ToLower(model), strings.ToLower(modelName)) {
			return modelLimits.ContextLimit, modelLimits.MaxAllowed
		}
	}

	// Default limits
	return 128000, 128000
}

// TruncationResult holds the result of truncation
type TruncationResult struct {
	Messages     []Message
	RemovedCount int
}

// TruncateConversation truncates conversation history
func TruncateConversation(messages []Message, strategy string) TruncationResult {
	if len(messages) <= 1 {
		return TruncationResult{Messages: messages, RemovedCount: 0}
	}

	// Keep system prompt (index 0)
	systemPrompt := messages[0]
	conversation := messages[1:]

	var keepCount int
	switch strategy {
	case "quarter":
		keepCount = len(conversation) / 4
	case "lastTwo", "last2":
		keepCount = 2
	case "half", "default":
		fallthrough
	default:
		keepCount = len(conversation) / 2
	}

	// Ensure at least 2 messages are kept
	if keepCount < 2 {
		keepCount = minInt(2, len(conversation))
	}

	truncated := conversation[len(conversation)-keepCount:]
	result := append([]Message{systemPrompt}, truncated...)

	return TruncationResult{
		Messages:     result,
		RemovedCount: len(conversation) - keepCount,
	}
}

// TruncateFileContent truncates file content if too large
func TruncateFileContent(content string, maxLines int) string {
	lines := strings.Split(content, "\n")
	if len(lines) <= maxLines {
		return content
	}
	// Keep first and last portions
	keepStart := maxLines / 2
	keepEnd := maxLines - keepStart
	return strings.Join(append(lines[:keepStart], lines[len(lines)-keepEnd:]...), "\n")
}

// TruncateOutput truncates command output
func TruncateOutput(output string, maxLines int, keepStart, keepEnd int) string {
	lines := strings.Split(output, "\n")
	if len(lines) <= maxLines {
		return output
	}

	if len(lines) <= keepStart+keepEnd {
		return output
	}

	result := append(lines[:keepStart], lines[len(lines)-keepEnd:]...)
	return strings.Join(result, "\n")
}

// DuplicateDetector detects duplicate file reads
type DuplicateDetector struct {
	reads map[string]int // path -> message index
}

// NewDuplicateDetector creates a new duplicate detector
func NewDuplicateDetector() *DuplicateDetector {
	return &DuplicateDetector{
		reads: make(map[string]int),
	}
}

// WasReadBefore checks if a file was read before
func (dd *DuplicateDetector) WasReadBefore(path string) (int, bool) {
	idx, ok := dd.reads[path]
	return idx, ok
}

// RecordRead records a file read
func (dd *DuplicateDetector) RecordRead(path string, messageIndex int) {
	dd.reads[path] = messageIndex
}

// Clear clears the detector
func (dd *DuplicateDetector) Clear() {
	dd.reads = make(map[string]int)
}

// ReplaceOldReads replaces old file reads with a notice
func ReplaceOldReads(messages []Message, path string, currentIndex int) int {
	replaced := 0
	for i := range messages {
		if strings.Contains(messages[i].Content, "[Context ID:") &&
			strings.Contains(messages[i].Content, "file: "+path) {
			// Replace with notice
			notice := fmt.Sprintf("[File re-read at message %d - previous content available in context]", currentIndex)
			messages[i].Content = notice
			replaced++
		}
	}
	return replaced
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}