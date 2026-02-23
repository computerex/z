#!/usr/bin/env python3
"""Test XML suppression logic."""

import sys
import re

# Simulate the suppression logic
tool_names = {'read_file', 'write_to_file', 'execute_command', 'list_files'}
param_names = {'path', 'content', 'command', 'recursive'}

test_xml = """I'll read the file for you.

<read_file>
<path>README.md</path>
</read_file>"""

print("Test XML:", test_xml)
print("\nProcessing...")

_suppressing = None
_param_suppressing = None
_tag_buf = ""
_in_tag = False

for c in test_xml:
    if _param_suppressing:
        _tag_buf += c
        close = f"</{_param_suppressing}>"
        if _tag_buf.endswith(close):
            _param_suppressing = None
            _tag_buf = ""
        continue
    
    if _suppressing:
        _tag_buf += c
        close = f"</{_suppressing}>"
        if _tag_buf.endswith(close):
            _suppressing = None
            _tag_buf = ""
        continue
    
    if _in_tag:
        _tag_buf += c
        if c == '>':
            m = re.match(r'</?(\w+)', _tag_buf)
            if m:
                tag_name = m.group(1)
                is_closing = _tag_buf.startswith('</')
                if tag_name in tool_names:
                    if not is_closing:
                        sys.stdout.write(f"\n[tool] {tag_name}")
                        _suppressing = tag_name
                        _tag_buf = ""
                    else:
                        _tag_buf = ""
                    _in_tag = False
                    continue
                elif tag_name in param_names:
                    if not is_closing:
                        _param_suppressing = tag_name
                        _tag_buf = ""
                    else:
                        _tag_buf = ""
                    _in_tag = False
                    continue
                else:
                    sys.stdout.write(_tag_buf)
            else:
                sys.stdout.write(_tag_buf)
            _in_tag = False
            continue
        if len(_tag_buf) > 80:
            sys.stdout.write(_tag_buf)
            _in_tag = False
            continue
    
    if c == '<':
        _in_tag = True
        _tag_buf = c
        continue
    
    sys.stdout.write(c)

print("\n\nDone!")