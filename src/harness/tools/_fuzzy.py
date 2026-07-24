"""Fuzzy matching helpers for replace_in_file — extracted from ToolHandlers."""

import difflib
import re
import unicodedata

from ..logger import get_logger

log = get_logger("tools")


def normalize_trailing(s: str) -> str:
    """Normalize line endings and trailing whitespace."""
    lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    return '\n'.join(line.rstrip() for line in lines)


def normalize_unicode(s: str) -> str:
    """Normalize Unicode substitutions LLMs commonly make.
    
    Ported from pi-mono's normalizeForFuzzyMatch:
    - NFKC normalization
    - Strip trailing whitespace per line
    - Smart quotes → ASCII equivalents
    - Unicode dashes/hyphens → hyphen-minus
    - Non-breaking and special spaces → regular space
    """
    s = unicodedata.normalize('NFKC', s)
    s = '\n'.join(line.rstrip() for line in s.replace('\r\n', '\n').split('\n'))
    s = re.sub(r'[\u2018\u2019\u201A\u201B]', "'", s)   # smart single quotes
    s = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', s)   # smart double quotes
    s = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]', '-', s)  # dashes
    s = re.sub(r'[\u00A0\u2002-\u200A\u202F\u205F\u3000]', ' ', s)  # special spaces
    return s


def unescape_text(s: str) -> str:
    """Normalize literal escape sequences (\\n → newline, \\t → tab etc.)."""
    return (s.replace('\\n', '\n')
             .replace('\\t', '\t')
             .replace('\\r', '\r')
             .replace("\\'", "'")
             .replace('\\"', '"'))


def strip_indent(s: str) -> list:
    """Return (stripped_lines, indent_per_line) for indentation-agnostic compare."""
    lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    stripped = [line.lstrip() for line in lines]
    indents = [line[:len(line) - len(line.lstrip())] for line in lines]
    return stripped, indents


def block_anchor_match(content_lines: list, search_lines: list):
    """Find block by anchoring on first+last lines, fuzzy-match middle.
    
    Ported from opencode's BlockAnchorReplacer + BlockAnchorReplacer logic.
    Returns (start_idx, end_exclusive_idx) or None.
    """
    # Strip trailing empty lines (like opencode's BlockAnchorReplacer)
    while search_lines and search_lines[-1].strip() == "":
        search_lines = search_lines[:-1]
    if len(search_lines) < 3:
        return None
    first = search_lines[0].strip()
    last = search_lines[-1].strip()
    if not first or not last:
        return None

    candidates = []
    for i, line in enumerate(content_lines):
        if line.strip() != first:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last:
                candidates.append((i, j))
                break  # only first occurrence of last line after this first

    if not candidates:
        return None

    def middle_similarity(cand_start: int, cand_end: int) -> float:
        s_mid = search_lines[1:-1]
        a_mid = content_lines[cand_start + 1:cand_end]
        n = max(len(s_mid), len(a_mid))
        if n == 0:
            return 1.0
        total = 0.0
        for s_line, a_line in zip(s_mid, a_mid):
            max_len = max(len(s_line.strip()), len(a_line.strip()))
            if max_len == 0:
                continue
            total += difflib.SequenceMatcher(None, s_line.strip(), a_line.strip()).ratio()
        return total / n

    if len(candidates) == 1:
        start, end = candidates[0]
        # Single candidate: accept if anchors match (threshold 0.0, like opencode)
        return start, end + 1

    # Multiple candidates: pick the one with highest middle similarity
    best = max(candidates, key=lambda c: middle_similarity(c[0], c[1]))
    start, end = best
    if middle_similarity(start, end) >= 0.3:
        return start, end + 1
    return None


def find_best_fuzzy_match(content_text: str, search_text: str):
    """Find the best fuzzy match for search_text within content_text.
    
    Returns (start_line_idx, end_line_idx, similarity_ratio) or None.
    
    Capped at 30 search lines to avoid O(n³) blowup on large blocks
    (e.g. model sends entire <style> block as old_text).  Large blocks
    should always be found by exact / whitespace / indent strategies.
    """
    search_lines = search_text.replace('\r\n', '\n').split('\n')
    content_lines = content_text.replace('\r\n', '\n').split('\n')
    search_len = len(search_lines)
    
    if search_len == 0 or len(content_lines) == 0:
        return None
    
    # Skip fuzzy matching for large blocks — too expensive and
    # unlikely to produce a useful result.
    if search_len > 30:
        return None
    
    best_ratio = 0.0
    best_start = 0
    best_window = search_len
    
    # Slide a window of size search_len (±30%) across content_lines
    min_window = max(1, int(search_len * 0.7))
    max_window = int(search_len * 1.3) + 1
    
    # Pre-filter threshold: at least 40% of lines must match exactly
    min_shared = max(2, int(search_len * 0.4))
    
    for window_size in range(min_window, min(max_window, len(content_lines) + 1)):
        for i in range(len(content_lines) - window_size + 1):
            candidate = content_lines[i:i + window_size]
            # Quick pre-filter: require a meaningful fraction of exact-match lines
            shared = sum(1 for a, b in zip(search_lines, candidate)
                         if a.strip() == b.strip())
            if shared < min_shared:
                continue
            
            ratio = difflib.SequenceMatcher(
                None,
                '\n'.join(search_lines),
                '\n'.join(candidate),
            ).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_window = window_size
    
    if best_ratio > 0.4:
        return best_start, best_start + best_window, best_ratio
    return None


def build_diagnostic(content_text: str, search_text: str) -> str:
    """Build a helpful error message showing the closest match."""
    match_info = find_best_fuzzy_match(content_text, search_text)
    
    search_lines = search_text.replace('\r\n', '\n').split('\n')
    content_lines = content_text.replace('\r\n', '\n').split('\n')
    
    msg_parts = [
        f"Error: old_text not found in file.",
        f"",
        f"old_text ({len(search_lines)} lines):",
        f"  {chr(10).join('  ' + l for l in search_lines[:5])}",
    ]
    if len(search_lines) > 5:
        msg_parts.append(f"  ... ({len(search_lines) - 5} more lines)")
    
    if match_info:
        start, end, ratio = match_info
        msg_parts.append(f"")
        msg_parts.append(f"Closest match in file (lines {start+1}-{end}, {ratio:.0%} similar):")
        for i in range(start, min(end, start + 10)):
            msg_parts.append(f"  {i+1:4d} | {content_lines[i]}")
        if end - start > 10:
            msg_parts.append(f"  ... ({end - start - 10} more lines)")
        msg_parts.append(f"")
        msg_parts.append(f"Tip: Re-read the file around lines {start+1}-{end} and retry with the exact content.")
    else:
        first_search = search_lines[0].strip() if search_lines else ""
        if first_search:
            near = [i for i, l in enumerate(content_lines) if first_search in l]
            if near:
                msg_parts.append(f"")
                msg_parts.append(f"First line of old_text found near line(s): {', '.join(str(n+1) for n in near[:5])}")
                ctx_start = max(0, near[0] - 2)
                ctx_end = min(len(content_lines), near[0] + 5)
                for i in range(ctx_start, ctx_end):
                    msg_parts.append(f"  {i+1:4d} | {content_lines[i]}")
        msg_parts.append(f"")
        msg_parts.append(f"Tip: Use read_file with start_line/end_line to see the exact content, then retry.")
    
    return '\n'.join(msg_parts)
