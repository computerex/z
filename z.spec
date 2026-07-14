# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os

# Use spec-relative paths so the config works across environments
_spec_dir = os.path.dirname(os.path.abspath(SPECPATH))
_src_dir = os.path.join(_spec_dir, 'src')

datas = [(_src_dir, 'src')]
binaries = []
hiddenimports = [
    'harness',
    'harness.cost_tracker',
    'harness.cron_scheduler',
    'harness.cron_tasks',
    'harness.cron_tool_handlers',
    'harness.streaming_client',
    'harness.tool_handlers',
    'harness.tool_registry',
    'harness.cline_agent',
    'harness.prompts',
    'harness.interrupt',
    'harness.config',
    'harness.context_management',
    'harness.smart_context',
    'harness.sub_agent_manager',
    'harness.hooks',
    'harness.memdir',
    'harness.plugin_manager',
    'harness.status_line',
    'harness.todo_manager',
    'harness.oauth',
    'harness.codex_oauth_client',
    'harness.copilot_oauth_client',
    'harness.cross_encoder',
    'harness.image_utils',
    'harness.model_capabilities',
    'harness.output_protocol',
    'harness.remote',
    'litellm',
    'litellm.llms',
    'mcp',
    'mcp.client.stdio',
    'mcp.client.sse',
    'mcp.client.streamable_http',
    'prompt_toolkit',
]
tmp_ret = collect_all('litellm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('prompt_toolkit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('rich')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

_harness_py = os.path.join(_spec_dir, 'harness.py')

a = Analysis(
    [_harness_py],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'PIL', 'numpy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='z',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='z',
)
