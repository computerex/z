import re

path = r'C:\Users\mohd\Downloads\Confidentiality and Work Product Agrmt_Mohd_v1.doc'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    rtf = f.read()

# Strip RTF headers and control words
# Remove RTF groups and control sequences
text = rtf

# Remove RTF font table, color table, stylesheet, etc.
text = re.sub(r'\{\\fonttbl[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\colortbl[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\stylesheet[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\list[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\listoverride[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\\*\\generator[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\\*\\xmlnstbl[^}]*\}', '', text, flags=re.DOTALL)
text = re.sub(r'\{\\\*\\latentstyles[^}]*\}', '', text, flags=re.DOTALL)

# Remove RTF control words (backslash + word)
text = re.sub(r'\\([a-z]+)(-?\d+)?', '', text)
# Remove special characters
text = re.sub(r'\\[{}]', '', text)
text = re.sub(r'\\\'[0-9a-fA-F]{2}', '', text)
text = re.sub(r'\\u\d+', '', text)
text = re.sub(r'\\[^a-z]', '', text)

# Remove braces
text = text.replace('{', '').replace('}', '')

# Remove multiple newlines and spaces
text = re.sub(r'\n{3,}', '\n\n', text)
text = re.sub(r' {2,}', ' ', text)

# Print
lines = [l.strip() for l in text.split('\n') if l.strip()]
for l in lines:
    print(l)
