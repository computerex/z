import re, os, sys

# Only scan the source directories we care about
files = []
for d in ['src/harness']:
    for root, dirs, fnames in os.walk(d):
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__') and d != 'node_modules']
        for f in fnames:
            if f.endswith('.py'):
                files.append(os.path.join(root, f))

# Also harness.py itself
if os.path.exists('harness.py'):
    files.append('harness.py')

imports = set()
for f in files:
    try:
        with open(f, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                m = re.match(r'^(?:import|from)\s+(\w+)', line)
                if m:
                    name = m.group(1)
                    if name.startswith('src') or name.startswith('.'):
                        continue
                    imports.add(name)
    except:
        pass

stdlib = {
    'abc','ast','asyncio','base64','collections','concurrent','contextlib','ctypes',
    'dataclasses','datetime','decimal','difflib','enum','faulthandler','functools',
    'hashlib','html','http','importlib','inspect','io','itertools','json','logging',
    'math','mimetypes','multiprocessing','operator','os','pathlib','platform','pprint',
    'queue','random','re','select','shlex','shutil','signal','socket','sqlite3',
    'statistics','string','struct','subprocess','sys','tempfile','textwrap','threading',
    'time','traceback','types','typing','unicodedata','urllib','uuid','warnings',
    'weakref','xml','zipfile','csv','copy','fnmatch','calendar','ssl','bisect',
    'argparse','codecs','filecmp','glob','gzip','locale','numbers','optparse',
    'pickle','plistlib','py_compile','secrets','shelve','site','tarfile','token',
    'tokenize','tty','unittest','venv','winreg','winsound','zipimport','zlib',
    '__future__','builtins','errno','gc','mmap','msvcrt','ntpath','posixpath',
    'pyexpat','configparser','runpy','__main__','grp','pwd','ensurepip',
    'compileall','contextvars','dis','keyword','linecache','symbol','fcntl',
    'resource','pdb','cProfile','profile','pstats','trace',
    'webbrowser','socketserver','nntplib','imaplib','mailbox','mailcap',
    'smtpd','smtplib','telnetlib','uu','xdrlib','nis','wave','sndhdr',
    'rlcompleter','reprlib','tabnanny','sunau','spwd','pty','pipes',
    'modulefinder','threading','dummy_threading',
    'sched','stat','test',
}

third = sorted(i for i in imports if i not in stdlib)
print('\n'.join(third))
print(f'\n--- COUNT: {len(third)}')
