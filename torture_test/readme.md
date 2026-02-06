# Tool Documentation

This document contains examples of XML tool usage for reference.

## read_file Tool

The `read_file` tool is used to read the contents of a file at a specified path.

**Example:**
```xml
<read_file>
<path>example.txt</path>
</read_file>
```

**Parameters:**
- `path` (required): The file path to read

## write_to_file Tool

The `write_to_file` tool is used to create new files with specified content.

**Example:**
```xml
<write_to_file>
<path>newfile.py</path>
<content>
def hello():
    print("Hello, World!")
</content>
</write_to_file>
```

**Parameters:**
- `path` (required): The file path to write
- `content` (required): The content to write to the file

## replace_in_file Tool

The `replace_in_file` tool is used to modify existing files using SEARCH/REPLACE blocks.

**Example:**
```xml
<replace_in_file>
<path>existing.py</path>
<diff>
<<<<<<< SEARCH
def old_function():
    return "old"
=======
def new_function():
    return "new"
>>>>>>> REPLACE
</diff>
</replace_in_file>
```

**Parameters:**
- `path` (required): The file path to modify
- `diff` (required): One or more SEARCH/REPLACE blocks defining exact changes

**Important Rules:**
1. SEARCH content must match exactly
2. Each block replaces only the first occurrence
3. Use multiple blocks for multiple changes
4. Keep blocks concise and unique

---

*Note: All examples shown above are for documentation purposes only.*