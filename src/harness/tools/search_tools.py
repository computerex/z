"""Search tools for the harness - lexical and semantic search."""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchMatch:
    """A search match result."""
    
    file_path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int
    context_before: List[str]
    context_after: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


async def lexical_search(
    directory: str,
    query: str,
    file_pattern: str = "*",
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = 50,
    context_lines: int = 2,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    """Perform lexical (text) search across files.
    
    Args:
        directory: Root directory to search.
        query: Search query string or regex pattern.
        file_pattern: Glob pattern for files to search (e.g., "*.py").
        is_regex: Whether query is a regex pattern.
        case_sensitive: Whether search is case-sensitive.
        max_results: Maximum number of results.
        context_lines: Number of context lines before/after match.
        include_hidden: Whether to include hidden files/directories.
    
    Returns:
        List of search match dictionaries.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Compile regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if is_regex:
        pattern = re.compile(query, flags)
    else:
        pattern = re.compile(re.escape(query), flags)
    
    results: List[SearchMatch] = []
    
    # Walk directory
    for root, dirs, files in os.walk(path):
        # Skip hidden directories
        if not include_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for file_name in files:
            # Skip hidden files
            if not include_hidden and file_name.startswith("."):
                continue
            
            # Check file pattern
            from fnmatch import fnmatch
            if not fnmatch(file_name, file_pattern):
                continue
            
            file_path = Path(root) / file_name
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    for match in pattern.finditer(line):
                        # Get context lines
                        start_ctx = max(0, i - context_lines)
                        end_ctx = min(len(lines), i + context_lines + 1)
                        
                        result = SearchMatch(
                            file_path=str(file_path),
                            line_number=i + 1,
                            line_content=line.rstrip(),
                            match_start=match.start(),
                            match_end=match.end(),
                            context_before=[l.rstrip() for l in lines[start_ctx:i]],
                            context_after=[l.rstrip() for l in lines[i + 1:end_ctx]],
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            return [r.to_dict() for r in results]
            
            except (PermissionError, OSError):
                continue
    
    return [r.to_dict() for r in results]


class SemanticIndex:
    """Semantic search index using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index: List[Dict[str, Any]] = []
        self._embeddings = None
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic search. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model
    
    def index_file(self, file_path: str, chunk_size: int = 500) -> int:
        """Index a file for semantic search.
        
        Args:
            file_path: Path to the file to index.
            chunk_size: Size of text chunks in characters.
        
        Returns:
            Number of chunks indexed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        # Split into chunks
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            line_with_newline = line + "\n"
            if current_size + len(line_with_newline) > chunk_size and current_chunk:
                chunks.append({
                    "file_path": str(path),
                    "start_line": start_line,
                    "end_line": i - 1,
                    "content": "".join(current_chunk),
                })
                current_chunk = [line_with_newline]
                current_size = len(line_with_newline)
                start_line = i
            else:
                current_chunk.append(line_with_newline)
                current_size += len(line_with_newline)
        
        if current_chunk:
            chunks.append({
                "file_path": str(path),
                "start_line": start_line,
                "end_line": len(lines),
                "content": "".join(current_chunk),
            })
        
        # Add to index
        self._index.extend(chunks)
        
        # Invalidate embeddings cache
        self._embeddings = None
        
        return len(chunks)
    
    def index_directory(
        self,
        directory: str,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ) -> int:
        """Index all files in a directory.
        
        Args:
            directory: Directory to index.
            file_patterns: List of glob patterns for files to include.
            exclude_patterns: List of glob patterns for files to exclude.
        
        Returns:
            Total number of chunks indexed.
        """
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.json"]
        if exclude_patterns is None:
            exclude_patterns = ["node_modules/*", "__pycache__/*", ".git/*", "*.pyc"]
        
        from fnmatch import fnmatch
        
        path = Path(directory)
        total_chunks = 0
        
        for root, dirs, files in os.walk(path):
            # Filter out excluded directories
            rel_root = Path(root).relative_to(path)
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch(str(rel_root / d), p) for p in exclude_patterns)
                and not d.startswith(".")
            ]
            
            for file_name in files:
                file_path = Path(root) / file_name
                rel_path = file_path.relative_to(path)
                
                # Check if file matches include patterns
                if not any(fnmatch(file_name, p) for p in file_patterns):
                    continue
                
                # Check if file matches exclude patterns
                if any(fnmatch(str(rel_path), p) for p in exclude_patterns):
                    continue
                
                try:
                    total_chunks += self.index_file(str(file_path))
                except (PermissionError, OSError, UnicodeDecodeError):
                    continue
        
        return total_chunks
    
    def _ensure_embeddings(self):
        """Ensure embeddings are computed."""
        if self._embeddings is None and self._index:
            model = self._get_model()
            texts = [chunk["content"] for chunk in self._index]
            self._embeddings = model.encode(texts, convert_to_numpy=True)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search the index semantically.
        
        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            min_score: Minimum similarity score (0-1).
        
        Returns:
            List of matching chunks with scores.
        """
        if not self._index:
            return []
        
        self._ensure_embeddings()
        
        model = self._get_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        import numpy as np
        
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            
            chunk = self._index[idx].copy()
            chunk["score"] = score
            results.append(chunk)
        
        return results
    
    def clear(self):
        """Clear the index."""
        self._index = []
        self._embeddings = None


# Global semantic index instance
_semantic_index: Optional[SemanticIndex] = None


def get_semantic_index() -> SemanticIndex:
    """Get or create the global semantic index."""
    global _semantic_index
    if _semantic_index is None:
        _semantic_index = SemanticIndex()
    return _semantic_index


async def semantic_search(
    query: str,
    directory: Optional[str] = None,
    top_k: int = 10,
    rebuild_index: bool = False,
) -> List[Dict[str, Any]]:
    """Perform semantic search across indexed files.
    
    Args:
        query: Natural language search query.
        directory: Directory to search (indexes if not already indexed).
        top_k: Maximum number of results.
        rebuild_index: Whether to rebuild the index.
    
    Returns:
        List of matching code chunks with relevance scores.
    """
    index = get_semantic_index()
    
    # Index directory if provided and needed
    if directory and (rebuild_index or not index._index):
        index.clear()
        index.index_directory(directory)
    
    return index.search(query, top_k=top_k)
