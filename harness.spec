# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['harness.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src')],
    hiddenimports=['harness', 'harness.cline_agent', 'harness.config', 'harness.context_management', 'harness.cost_tracker', 'harness.interrupt', 'harness.prompts', 'harness.streaming_client', 'harness.tool_handlers', 'harness.todo_manager', 'harness.smart_context', 'harness.workspace_index', 'harness.logger', 'harness.status_line', 'httpx', 'httpx._transports', 'httpx._transports.default', 'pydantic', 'pydantic.main', 'pydantic._internal', 'pydantic_core', 'tiktoken', 'tiktoken_ext', 'tiktoken_ext.openai_public', 'rich', 'rich.console', 'rich.panel', 'rich.markup', 'rich.tree', 'rich.text', 'prompt_toolkit', 'prompt_toolkit.key_binding', 'prompt_toolkit.keys', 'prompt_toolkit.history', 'prompt_toolkit.auto_suggest', 'prompt_toolkit.formatted_text', 'asyncio', 'json', 're', 'pathlib', 'encodings.utf_8', 'encodings.cp1252', 'encodings.ascii', 'aiofiles', 'anyio', 'sniffio', 'h11', 'httpcore', 'certifi', 'psutil'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'pytest_asyncio'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='harness',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
