package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// WorkspaceIndex holds the project file structure.
type WorkspaceIndex struct {
	Root      string
	Files     []FileInfo
	BuildTime time.Duration
	IsGit     bool
}

// FileInfo describes a single file.
type FileInfo struct {
	RelPath   string
	Size      int64
	Extension string
	Language  string
	IsBinary  bool
}

var skipDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true, "node_modules": true,
	"__pycache__": true, ".venv": true, "venv": true, "env": true,
	".env": true, "dist": true, "build": true, "target": true,
	"vendor": true, "obj": true, "bin": true, ".tox": true,
	".mypy_cache": true, ".pytest_cache": true, ".ruff_cache": true,
	".next": true, ".nuxt": true, ".output": true, "coverage": true,
	".coverage": true, ".idea": true, ".vscode": true,
	".harness_output": true, ".sessions": true,
}

var binaryExts = map[string]bool{
	".exe": true, ".dll": true, ".so": true, ".dylib": true, ".o": true,
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".bmp": true,
	".ico": true, ".svg": true, ".webp": true,
	".mp3": true, ".mp4": true, ".wav": true, ".ogg": true,
	".zip": true, ".tar": true, ".gz": true, ".bz2": true, ".xz": true,
	".7z": true, ".rar": true,
	".pdf": true, ".doc": true, ".docx": true,
	".bin": true, ".dat": true, ".db": true, ".sqlite": true,
	".woff": true, ".woff2": true, ".ttf": true, ".otf": true,
	".pyc": true, ".pyo": true, ".class": true, ".jar": true,
	".model": true, ".onnx": true, ".pt": true, ".pth": true,
}

var langMap = map[string]string{
	".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
	".go": "Go", ".rs": "Rust", ".java": "Java", ".c": "C", ".cpp": "C++",
	".h": "C", ".hpp": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
	".swift": "Swift", ".lua": "Lua", ".sh": "Shell", ".ps1": "PowerShell",
	".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
	".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
	".xml": "XML", ".md": "Markdown", ".proto": "Protobuf",
	".vue": "Vue", ".svelte": "Svelte", ".jsx": "JavaScript",
}

// BuildWorkspaceIndex scans the workspace for files.
func BuildWorkspaceIndex(root string) *WorkspaceIndex {
	t0 := time.Now()
	idx := &WorkspaceIndex{Root: root}

	// Try git ls-files first
	files := gitLsFiles(root)
	if files != nil {
		idx.IsGit = true
		logInfo("Index: using git ls-files (%d files)", len(files))
	} else {
		files = smartWalk(root)
		logInfo("Index: smart walk (%d files)", len(files))
	}

	for _, relPath := range files {
		ext := strings.ToLower(filepath.Ext(relPath))
		fi := FileInfo{
			RelPath:   filepath.ToSlash(relPath),
			Extension: ext,
			Language:  langMap[ext],
			IsBinary:  binaryExts[ext],
		}
		absPath := filepath.Join(root, relPath)
		if stat, err := os.Stat(absPath); err == nil {
			fi.Size = stat.Size()
		}
		idx.Files = append(idx.Files, fi)
	}

	idx.BuildTime = time.Since(t0)
	logInfo("Index built: %d files in %.2fs", len(idx.Files), idx.BuildTime.Seconds())
	return idx
}

// CompactTree returns a tree-style summary for the system prompt.
func (idx *WorkspaceIndex) CompactTree() string {
	if len(idx.Files) == 0 {
		return "(empty workspace)"
	}

	// Group by top-level directory
	dirs := map[string][]string{}
	var rootFiles []string

	for _, f := range idx.Files {
		parts := strings.SplitN(f.RelPath, "/", 2)
		if len(parts) == 1 {
			rootFiles = append(rootFiles, f.RelPath)
		} else {
			dirs[parts[0]] = append(dirs[parts[0]], parts[1])
		}
	}

	var sb strings.Builder
	sb.WriteString(filepath.Base(idx.Root) + "/\n")

	// Root files first
	sort.Strings(rootFiles)
	for _, f := range rootFiles {
		if len(rootFiles) > 30 {
			// Truncate
			sb.WriteString(fmt.Sprintf("  %s\n", f))
			if f == rootFiles[29] {
				sb.WriteString(fmt.Sprintf("  ... and %d more files\n", len(rootFiles)-30))
				break
			}
		} else {
			sb.WriteString(fmt.Sprintf("  %s\n", f))
		}
	}

	// Directories
	dirNames := make([]string, 0, len(dirs))
	for d := range dirs {
		dirNames = append(dirNames, d)
	}
	sort.Strings(dirNames)

	for _, d := range dirNames {
		files := dirs[d]
		if len(files) <= 8 {
			sb.WriteString(fmt.Sprintf("  %s/\n", d))
			for _, f := range files {
				sb.WriteString(fmt.Sprintf("    %s\n", f))
			}
		} else {
			sb.WriteString(fmt.Sprintf("  %s/ (%d files)\n", d, len(files)))
			// Show first 5
			for i := 0; i < 5 && i < len(files); i++ {
				sb.WriteString(fmt.Sprintf("    %s\n", files[i]))
			}
			sb.WriteString(fmt.Sprintf("    ... and %d more\n", len(files)-5))
		}
	}

	return sb.String()
}

func gitLsFiles(root string) []string {
	cmd := exec.Command("git", "ls-files", "--cached", "--others", "--exclude-standard")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var files []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		absPath := filepath.Join(root, line)
		if info, err := os.Stat(absPath); err == nil && !info.IsDir() {
			files = append(files, line)
		}
	}
	return files
}

func smartWalk(root string) []string {
	var files []string
	maxFiles := 10000

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if len(files) >= maxFiles {
			return filepath.SkipDir
		}

		rel, _ := filepath.Rel(root, path)
		if rel == "." {
			return nil
		}

		name := info.Name()

		if info.IsDir() {
			if skipDirs[name] || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}

		if strings.HasPrefix(name, ".") {
			return nil
		}

		files = append(files, rel)
		return nil
	})

	return files
}
