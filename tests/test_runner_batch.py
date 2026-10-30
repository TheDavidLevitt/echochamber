"""End-to-end runner and batch config loading."""

import json

from helpers import FakeProvider

from echochamber.batch import RunVariation, load_config
from echochamber.core.runner import DebateSpec, run_debate


def fake_factory_from(fakes):
    """provider_factory that hands out fakes in creation order (P, D, M)."""
    queue = list(fakes)

    def factory(provider, model=None, **kwargs):
        return queue.pop(0)

    return factory


def test_run_debate_end_to_end(tmp_path):
    factory = fake_factory_from([
        FakeProvider(["prosecution argument"]),
        FakeProvider(["defense argument"]),
        FakeProvider([
            "opening statement",
            "CONTINUE: YES\nWINNER: NONE\nREASONING: even",
            "Summary. FINAL VERDICT: DRAW",
        ]),
    ])
    spec = DebateSpec(
        topic="Test topic",
        position="Test position",
        max_rounds=1,
        enable_search=False,
        verbose=False,
        transcript_dir=str(tmp_path / "transcripts"),
        usage_log=str(tmp_path / "usage.jsonl"),
    )

    outcome = run_debate(spec, provider_factory=factory)

    assert outcome.result.winner == "draw"
    assert outcome.transcript_path is not None and outcome.transcript_path.exists()
    assert "Test topic" in outcome.transcript_path.read_text()

    # Every provider call was metered and exported
    events = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(events) == 5  # opening + P + D + eval + ruling
    assert {e["module"] for e in events} == {
        "debate.prosecution", "debate.defense", "debate.moderator",
    }


def test_load_config_merges_base(tmp_path):
    config = tmp_path / "runs.yaml"
    config.write_text("""
base:
  topic: "Base topic"
  position: "Base position"
  max_rounds: 2

runs:
  - name: "run-one"
    prosecution_provider: "anthropic"
  - name: "run-two"
    max_rounds: 5
""")
    variations = load_config(str(config))

    assert [v.name for v in variations] == ["run-one", "run-two"]
    assert variations[0].topic == "Base topic"
    assert variations[0].prosecution_provider == "anthropic"
    assert variations[0].max_rounds == 2
    assert variations[1].max_rounds == 5  # run overrides base


def test_variation_to_spec_is_quiet():
    var = RunVariation(name="x", topic="t", position="p", moderator_provider="together")
    spec = var.to_spec(output_dir=__import__("pathlib").Path("/tmp/out"))
    assert spec.verbose is False
    assert spec.moderator_model == "deepseek-ai/DeepSeek-R1"  # role-specific default
