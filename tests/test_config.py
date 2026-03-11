from pathlib import Path

from harness import config as cfg


def test_workspace_config_is_ignored_global_config_is_used(tmp_path, monkeypatch):
    global_cfg = tmp_path / "global.json"
    ws_cfg = tmp_path / "workspace.json"

    global_cfg.write_text(
        '{"api_url":"https://api.example.com/v1/","api_key":"k-global","model":"m-global","temperature":0.3}',
        encoding="utf-8",
    )
    ws_cfg.write_text(
        '{"api_url":"","api_key":"","model":"","max_tokens":64000}',
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg, "get_global_config_path", lambda: global_cfg)
    monkeypatch.setattr(cfg, "get_workspace_config_path", lambda workspace=None: ws_cfg)

    c = cfg.Config.from_json(workspace=Path("C:/tmp/ws"))
    assert c.api_url == "https://api.example.com/v1/"
    assert c.api_key == "k-global"
    assert c.model == "m-global"
    assert c.max_tokens == 128000


def test_runtime_overrides_still_override_global(tmp_path, monkeypatch):
    global_cfg = tmp_path / "global.json"
    ws_cfg = tmp_path / "workspace.json"

    global_cfg.write_text(
        '{"api_url":"https://api.example.com/v1/","api_key":"k-global","model":"m-global"}',
        encoding="utf-8",
    )
    ws_cfg.write_text(
        '{"api_url":"https://api.ws.example/v1/","api_key":"k-ws","model":"m-ws"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(cfg, "get_global_config_path", lambda: global_cfg)
    monkeypatch.setattr(cfg, "get_workspace_config_path", lambda workspace=None: ws_cfg)

    c = cfg.Config.from_json(
        workspace=Path("C:/tmp/ws"),
        overrides={
            "api_url": "https://api.runtime.example/v1/",
            "api_key": "k-runtime",
            "model": "m-runtime",
        },
    )
    assert c.api_url == "https://api.runtime.example/v1/"
    assert c.api_key == "k-runtime"
    assert c.model == "m-runtime"
