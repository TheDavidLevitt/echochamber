"""GUI setup-tab helpers: key masking, saving, and status overview."""

import os
import stat

import pytest

from echochamber import ui


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(ui, "ENV_FILE", path)
    for key in ui.PROVIDER_ENV_KEYS.values():
        monkeypatch.delenv(key, raising=False)
    return path


def test_mask_hides_key_material():
    masked = ui._mask("sk-ant-api03-averylongsecretkeyvalue1234")
    assert "averylongsecret" not in masked
    assert masked.startswith("set (sk-ant-")
    assert ui._mask("") == "not set"


def test_save_keys_writes_env_with_restrictive_perms(env_file):
    overview, status = ui.save_keys("sk-ant-test123", "", "", "AIza-test456")

    content = env_file.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-test123" in content
    assert "GEMINI_API_KEY=AIza-test456" in content
    assert "OPENAI_API_KEY" not in content  # empty fields untouched
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600
    assert "✅ Saved" in status
    # Live environment updated without restart
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test123"


def test_save_keys_tightens_preexisting_loose_env(env_file):
    env_file.write_text("OTHER=keepme\n")
    env_file.chmod(0o644)
    ui.save_keys("sk-ant-test123", "", "", "")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert "OTHER=keepme" in env_file.read_text()


def test_env_file_is_created_private_without_a_chmod(env_file, monkeypatch):
    """The mode is set when the file is created, not fixed up afterwards."""
    old_umask = os.umask(0o022)  # a typical umask: new files 0644 by default
    try:
        # If the implementation fell back to write-then-chmod, the file
        # would end up 0644 with Path.chmod disabled.
        monkeypatch.setattr(type(env_file), "chmod", lambda self, mode: None)
        ui.save_keys("", "", "", "AIza-test456")
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_save_keys_merges_existing_lines(env_file):
    env_file.write_text("# comment\nANTHROPIC_API_KEY=old-value\nOTHER=keepme\n")
    ui.save_keys("new-value", "", "", "")

    lines = env_file.read_text().splitlines()
    assert "ANTHROPIC_API_KEY=new-value" in lines
    assert "OTHER=keepme" in lines
    assert "# comment" in lines
    assert "old-value" not in env_file.read_text()


def test_save_keys_all_empty_is_noop(env_file):
    _, status = ui.save_keys("", "", "", "")
    assert "Nothing to save" in status
    assert not env_file.exists()


def test_overview_reports_masked_status(env_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-somethinglong12345")
    md = ui.key_overview_md()
    assert "not set" in md            # untouched providers
    assert "sk-ant-" in md            # masked prefix shown
    assert "somethinglong" not in md  # secret body hidden


def test_key_test_requires_a_key(env_file):
    assert ui.test_provider_key("anthropic", "").startswith("❌")


def test_run_debate_ui_tolerates_untouched_none_inputs():
    """Gradio sends None for never-touched Textboxes; first yield must not crash."""
    gen = ui.run_debate_ui(
        "Topic", "Position",
        "lmstudio", None, None,
        "lmstudio", None, None,
        "lmstudio", None, None,
        None, None,  # case folder + context strategy untouched
        1, None, False, None, None, None,  # iterations/budget/force/on_close untouched
    )
    status, tokens, chat, verdict = next(gen)
    assert "Starting debate" in status
    gen.close()


def test_save_text_evidence_writes_entry_files(tmp_path):
    import json as _json

    entries = _json.dumps([
        "Witness saw the defendant at 9pm",
        "Bank records:\n- transfer of $9,000\n- dated March 3",
    ])
    status = ui.save_text_evidence(entries, str(tmp_path / "case"), "prosecution")
    assert "✅ Saved 2 entries" in status

    files = sorted((tmp_path / "case" / "prosecution").glob("entry-*.txt"))
    assert len(files) == 2
    assert "Witness saw" in files[0].read_text()
    assert "- transfer of $9,000" in files[1].read_text()  # multi-line preserved
    # standard case-folder skeleton was created
    assert (tmp_path / "case" / "shared").is_dir()

    # appending later continues the numbering
    status2 = ui.save_text_evidence(_json.dumps(["third"]), str(tmp_path / "case"), "prosecution")
    assert "entry-03" in status2


def test_save_text_evidence_empty_is_noop(tmp_path):
    assert "Nothing to save" in ui.save_text_evidence("[]", str(tmp_path / "c"), "shared")
