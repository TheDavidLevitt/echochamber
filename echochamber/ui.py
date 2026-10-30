#!/usr/bin/env python3
"""
EchoChamber GUI - watch a debate live in the browser.

    uv run python -m echochamber.ui          # http://localhost:7860

Requires the ui extra:  uv sync --extra ui

Two tabs:
- Debate: run and watch debates (live agent/model status, token ticker,
  stop button, APA model recommendations).
- Setup: paste provider API keys, test them, and save to .env — no
  terminal needed.

Closing the tab while a debate runs triggers the browser's native leave
warning; whether the debate then aborts or continues in the background is a
user preference on the Debate tab (browsers do not allow custom dialogs on
tab close). Backgrounded debates appear in the "Background runs" panel and
still save their transcripts.

Set $ECHOCHAMBER_APA_DATA to an APA data directory (or merged export) to see
live procurement recommendations instead of the bundled sample.
"""

import json
import logging
import os
import queue
import re
import stat
import threading
from collections import Counter
from pathlib import Path

# The module stays importable without gradio (key helpers are GUI-independent
# and the test suite imports them); only building/launching the app needs it.
try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

_GRADIO_MISSING = "The GUI requires gradio. Install with: uv sync --extra ui"

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Package imports deliberately follow load_dotenv() so keys from .env are in
# the environment before any module reads them.
from .ablation import (  # noqa: E402
    AblationConfig,
    Scenario,
    format_report,
    run_ablation,
)
from .core.costs import estimate_debate_footprint, estimate_run_cost  # noqa: E402
from .core.evidence import EvidenceStore  # noqa: E402
from .core.runner import DebateSpec, resolve_model, run_debate  # noqa: E402
from .core.session import TerminationReason  # noqa: E402
from .core.usage import UsageMeter  # noqa: E402
from .recommendations import (  # noqa: E402
    credits_callout,
    funding_class,
    load_apa_data,
    recommend_for_debate,
)

logger = logging.getLogger(__name__)

PROVIDERS = ["lmstudio", "anthropic", "openai", "together", "gemini"]
DEFAULT_PROVIDER = os.environ.get("ECHOCHAMBER_DEFAULT_PROVIDER", "lmstudio")
ROLE_EMOJI = {"prosecution": "⚖️", "defense": "🛡️", "moderator": "👨‍⚖️", "system": "⚙️"}

# Source-of-funds palette (matches the AgentStable dashboard convention):
# red = out-of-pocket, green = finite credits, blue = subscription/included.
FUNDING_STYLE = {
    "real": ("#f87171", "paid API"),
    "credit": ("#4ade80", "credits"),
    "included": ("#60a5fa", "subscription"),
}


def _fund_class(provider: str) -> str:
    return funding_class(provider, _apa.credits if _apa else {})


def _fund_span(cls: str, amount: str, label: str = None) -> str:
    color, default_label = FUNDING_STYLE.get(cls, FUNDING_STYLE["real"])
    return (f"<span style='color:{color};font-weight:bold'>"
            f"{amount} {label or default_label}</span>")

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

ENV_FILE = Path.cwd() / ".env"

# The foreground run's stop event (single-user local app) and the run registry
# for the background panel.
_active_stop: dict = {"event": None}
_runs: list[dict] = []

# Warn (native browser dialog) when leaving while a debate is running, and
# initialize the text-evidence entry editors (contenteditable blocks where
# Enter starts a new entry — marked by a faint rule — and Shift+Enter
# continues the current entry on a new line).
_HEAD_JS = """
<style>
.ec-entry-host {
  min-height: 7em; max-height: 20em; overflow-y: auto;
  border: 1px solid rgba(128,128,128,.35); border-radius: 6px;
  padding: 8px 10px; font-size: .9em; line-height: 1.45; outline: none;
}
.ec-entry-host:focus { border-color: rgba(99,102,241,.8); }
.ec-entry { padding: 4px 0; min-height: 1.2em; }
.ec-entry + .ec-entry { border-top: 1px solid rgba(128,128,128,.28); }
.ec-entry-host:empty::before, .ec-entry:only-child:empty::before {
  content: 'Type evidence… Enter = new entry, Shift+Enter = new line';
  opacity: .45;
}
/* Mounted-but-invisible sync channel (visible=False would leave the DOM) */
.ec-hidden-sync { position: absolute !important; left: -9999px !important;
  height: 1px !important; width: 1px !important; overflow: hidden !important; }
</style>
<script>
window.__ec_running = false;
window.addEventListener('beforeunload', (e) => {
  if (window.__ec_running) { e.preventDefault(); e.returnValue = ''; }
});

function ecSync(host) {
  const entries = Array.from(host.querySelectorAll(':scope > .ec-entry'))
    .map((d) => d.innerText.replace(/\\u00a0/g, ' ').trim())
    .filter((t) => t.length > 0);
  const sync = document.querySelector('#' + host.dataset.sync + ' textarea');
  if (!sync) return;
  sync.value = JSON.stringify(entries);
  sync.dispatchEvent(new Event('input', { bubbles: true }));
}

function ecNewEntry(host) {
  const sel = window.getSelection();
  let current = sel.anchorNode;
  while (current && current.parentNode !== host) current = current.parentNode;
  const entry = document.createElement('div');
  entry.className = 'ec-entry';
  entry.innerHTML = '<br>';
  if (current && current.nextSibling) host.insertBefore(entry, current.nextSibling);
  else host.appendChild(entry);
  const range = document.createRange();
  range.setStart(entry, 0);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}

function ecInitEditors() {
  document.querySelectorAll('.ec-entry-host').forEach((host) => {
    if (host.dataset.ecInit) return;
    host.dataset.ecInit = '1';
    host.contentEditable = 'true';
    if (!host.querySelector('.ec-entry')) {
      host.innerHTML = '<div class="ec-entry"><br></div>';
    }
    host.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        ecNewEntry(host);
        ecSync(host);
      }
      // Shift+Enter: default behavior — a new line inside the current entry
    });
    host.addEventListener('input', () => ecSync(host));
  });
}
setInterval(ecInitEditors, 700);
window.ecClearEditor = function (hostId) {
  const host = document.getElementById(hostId);
  if (!host) return;
  host.innerHTML = '<div class="ec-entry"><br></div>';
  ecSync(host);
};
</script>
"""

# APA recommendations, loaded once at startup
try:
    _apa = load_apa_data()
    _recs = recommend_for_debate(_apa)
except Exception:
    # Startup guard: recommendations are optional and must never stop the
    # GUI from loading, whatever shape the external export is in.
    logger.warning("AgentStable recommendations unavailable", exc_info=True)
    _apa, _recs = None, {}


# ---------------------------------------------------------------- APA panel

def _recs_markdown() -> str:
    if not _recs or not any(_recs.values()):
        return "_No APA data available._"
    lines = []
    callout = credits_callout(_apa) if _apa else ""
    if callout:
        lines.append(callout)
        lines.append("")
    lines.append(f"_Source: {_apa.source}_\n")
    for role, items in _recs.items():
        lines.append(f"**{ROLE_EMOJI[role]} {role.capitalize()}**")
        for r in items:
            price = f"${r.price_in:g} / ${r.price_out:g} per 1M" if r.price_in is not None else "price unknown"
            price_tag = _fund_span(_fund_class(r.provider), price)
            lines.append(f"{r.rank}. `{r.model}` ({r.provider}) — {price_tag}" +
                         (f" · bar: {r.benchmark}" if r.benchmark else ""))
            lines.append(f"   ↳ {r.justification}")
            if r.funding_note:
                lines.append(f"   💰 {r.funding_note}")
        lines.append("")
    return "\n".join(lines)


_ESTIMATE_DISCLAIMER = (
    "  \n_Disclaimer: this is a rough approximation; the actual footprint may "
    "be higher. Setting a hard token budget is strongly recommended._"
)


def estimate_md(pros_provider, pros_model, def_provider, def_model,
                mod_provider, mod_model, rounds, iterations,
                token_budget=None) -> str:
    """Live footprint estimate. Money is bold red; warns if budget < estimate."""
    try:
        models = [
            resolve_model(pros_provider, _text(pros_model) or None, "advocate"),
            resolve_model(def_provider, _text(def_model) or None, "advocate"),
            resolve_model(mod_provider, _text(mod_model) or None, "moderator"),
        ]
        fp = estimate_debate_footprint(
            *models, max_rounds=int(rounds or 1), iterations=int(iterations or 1),
        )
    except Exception as e:
        # UI guard: the estimate is advisory and reruns on every input change.
        logger.debug("Debate estimate failed", exc_info=True)
        return f"_Estimate unavailable: {e}_"

    minutes = fp["seconds"] / 60
    time_str = f"~{minutes:.0f} min" if minutes >= 1 else f"~{fp['seconds']:.0f}s"

    # Split the money estimate by source of funds (dashboard convention:
    # red = paid API, green = credits, blue = subscription); if the roles
    # mix classes, each class estimate appears side by side.
    _, _, breakdown = estimate_run_cost(*models, max_rounds=int(rounds or 1))
    role_providers = {
        "prosecution": pros_provider, "defense": def_provider, "moderator": mod_provider,
    }
    per_class: dict = {}
    for role_key, data in breakdown.items():
        cls = _fund_class(role_providers[role_key])
        per_class[cls] = per_class.get(cls, 0.0) + data["cost"] * fp["iterations"]

    parts = []
    for cls in ("real", "credit", "included"):
        if cls not in per_class:
            continue
        mid = per_class[cls]
        if mid == 0:
            parts.append(_fund_span(cls, "free", "(local)"))
        else:
            parts.append(_fund_span(cls, f"${mid * 0.5:.2f}–${mid * 1.5:.2f}"))
    cost_str = " · ".join(parts) if parts else "free (local models)"
    real_max = per_class.get("real", 0.0) * 1.5
    flag = "🔴" if real_max > 1.0 else "🟢"
    lines = [
        f"**Estimated footprint:** {flag} ~{fp['tokens']:,} tokens · {cost_str} · "
        f"{time_str} ({fp['calls']} model calls, {fp['iterations']} iteration(s))"
        + _ESTIMATE_DISCLAIMER
    ]
    try:
        budget = int(token_budget) if token_budget else None
    except (TypeError, ValueError):
        budget = None
    if budget and fp["tokens"] > budget:
        lines.append(
            "<span style='color:#f87171;font-weight:bold'>⚠️ Warning: estimated "
            "token burn exceeds the hard limit; the verdict may not be rendered. "
            "Raise the budget or enable force-verdict.</span>"
        )
    return "\n\n".join(lines)


def _position_outcome(winner, position) -> str:
    if winner == "prosecution":
        return f"position upheld: “{position}”"
    if winner == "defense":
        return f"position rejected: “{position}”"
    return f"no decision reached on “{position}”"


def verdict_headline(winners: list, position: str) -> str:
    """Prominent verdict line, aggregated across iterations."""
    if not winners:
        return "## 🏛️ No verdict"
    if len(winners) == 1:
        w = winners[0] or "undecided"
        return f"## 🏛️ Verdict: **{w.upper()}** — {_position_outcome(w, position)}"
    tally = Counter(w or "undecided" for w in winners)
    top, n = tally.most_common(1)[0]
    detail = " · ".join(f"{w}: {c}" for w, c in tally.most_common())
    return (f"## 🏛️ Verdict across {len(winners)} iterations: **{top.upper()} {n}/{len(winners)}** "
            f"— {_position_outcome(top, position)}\n({detail})")


def _apply_recs(rank: int):
    """Provider/model values for all three roles from APA picks at the given rank."""
    out = []
    for role in ("prosecution", "defense", "moderator"):
        items = _recs.get(role) or []
        pick = next((r for r in items if r.rank == rank), items[-1] if items else None)
        if pick:
            out.extend([pick.provider, pick.model])
        else:
            out.extend([gr.skip(), gr.skip()])
    return out


# ------------------------------------------------------------- key handling

def _mask(value: str) -> str:
    if not value:
        return "not set"
    if len(value) <= 12:
        return "set (short key)"
    return f"set ({value[:7]}…{value[-4:]})"


def key_overview_md() -> str:
    lines = ["| Provider | Env var | Status |", "|---|---|---|"]
    for provider, env_key in PROVIDER_ENV_KEYS.items():
        lines.append(f"| {provider} | `{env_key}` | {_mask(os.environ.get(env_key, ''))} |")
    lines.append("| lmstudio | _(none needed)_ | local server on :1234 |")
    lines.append(
        "\nKeys are saved to `.env` in the folder EchoChamber was launched from "
        "(owner-read-only) and loaded on launch."
    )
    return "\n".join(lines)


def test_provider_key(provider: str, key: str) -> str:
    """Cheap auth check per provider (list-models style calls are free)."""
    key = (key or "").strip() or os.environ.get(PROVIDER_ENV_KEYS.get(provider, ""), "")
    try:
        if provider == "lmstudio":
            from .providers.lmstudio import get_lmstudio_model
            model = get_lmstudio_model()
            return f"✅ lmstudio: reachable, serving `{model}`" if model else \
                "❌ lmstudio: no server on localhost:1234 (start LM Studio and load a model)"
        if provider == "gemini":
            # Key OR Vertex env config both count as configured
            from .providers.gemini import build_client
            client, mode = build_client(key or None)
            next(iter(client.models.list()))
            return f"✅ gemini: works ({mode})"
        if not key:
            return f"❌ {provider}: no key entered or saved"
        if provider == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=key).models.list(limit=1)
        elif provider == "openai":
            from openai import OpenAI
            OpenAI(api_key=key).models.list()
        elif provider == "together":
            # Together's /models returns a bare array the OpenAI SDK can't
            # parse; a raw authenticated GET is the reliable check.
            import urllib.request
            req = urllib.request.Request(
                "https://api.together.xyz/v1/models",
                headers={
                    "Authorization": f"Bearer {key}",
                    # Together's edge rejects the default Python-urllib UA
                    "User-Agent": "echochamber-setup-check",
                },
            )
            urllib.request.urlopen(req, timeout=10)
        return f"✅ {provider}: key works"
    except ImportError as e:
        return f"❌ {provider}: SDK not installed ({e.name}) — run: uv sync --extra all"
    except Exception as e:
        # UI guard: any failure is the answer to "does this key work?"
        logger.debug("Key test failed for %s", provider, exc_info=True)
        return f"❌ {provider}: {str(e)[:200]}"


def save_keys(anthropic_key: str, openai_key: str, together_key: str, gemini_key: str):
    """Merge non-empty keys into .env (created owner-read-only) and the live env."""
    new_values = {
        "ANTHROPIC_API_KEY": anthropic_key.strip(),
        "OPENAI_API_KEY": openai_key.strip(),
        "TOGETHER_API_KEY": together_key.strip(),
        "GEMINI_API_KEY": gemini_key.strip(),
    }
    new_values = {k: v for k, v in new_values.items() if v}
    if not new_values:
        return key_overview_md(), "Nothing to save — all fields empty."

    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    for env_key, value in new_values.items():
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{env_key}="):
                lines[i] = f"{env_key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{env_key}={value}")
        os.environ[env_key] = value  # effective immediately, no restart

    ENV_FILE.write_text("\n".join(lines) + "\n")
    ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    saved = ", ".join(new_values)
    return key_overview_md(), f"✅ Saved {saved} to {ENV_FILE} (permissions 600). Empty fields were left unchanged."


# ------------------------------------------------------------ debate running

def _cost_spans(meter: UsageMeter) -> str:
    """Metered cost split by funding class, colored per the dashboard scheme."""
    totals: dict = {}
    for e in list(meter.events):
        cls = _fund_class(e.provider.split("/")[0])
        totals[cls] = totals.get(cls, 0.0) + e.cost_usd
    parts = [
        _fund_span(cls, f"${totals[cls]:.4f}")
        for cls in ("real", "credit", "included") if cls in totals
    ]
    return " + ".join(parts) if parts else "$0.0000"


def _fmt_tokens(meter: UsageMeter, budget: int | None) -> str:
    tin, tout = meter.total_tokens()
    total = tin + tout
    budget_str = f" / {budget:,} budget" if budget else ""
    pct = f" ({100 * total / budget:.0f}%)" if budget else ""
    return (f"**Tokens burned:** {total:,}{budget_str}{pct} · "
            f"{tin:,} in / {tout:,} out · **Cost:** {_cost_spans(meter)}")


def stop_debate():
    event = _active_stop["event"]
    if event:
        event.set()
        return "### ⏹️ Stopping after the current turn…"
    return "### 💤 No debate running."


def background_md() -> str:
    detached = [r for r in _runs if r["detached"]][-5:]
    if not detached:
        return "_No background runs._"
    lines = ["**Background runs** (tab was closed; debates kept going)\n"]
    for r in detached:
        tin, tout = r["meter"].total_tokens()
        lines.append(f"- *{r['topic']}* — {r['status']} · {tin + tout:,} tokens, "
                     f"${r['meter'].total_cost():.4f}" +
                     (f" · `{r['transcript']}`" if r["transcript"] else ""))
    return "\n".join(lines)


def inspect_evidence(path: str) -> str:
    """Preview what a case folder would load for each role."""
    path = (path or "").strip()
    if not path:
        return "_No case folder set — the debate runs on the topic alone._"
    try:
        from .core.evidence import EvidenceStore
        store = EvidenceStore.load(path)
        return f"```\n{store.summary()}\n```"
    except Exception as e:
        # UI guard: shown to the user in place of the preview.
        logger.debug("Case folder preview failed for %s", path, exc_info=True)
        return f"❌ Could not load `{path}`: {e}"


def _text(value) -> str:
    """Normalize an optional Textbox value — untouched fields arrive as None."""
    return (value or "").strip()


def run_debate_ui(topic, position, pros_provider, pros_model, pros_instr,
                  def_provider, def_model, def_instr,
                  mod_provider, mod_model, mod_instr,
                  case_folder, context_strategy,
                  rounds, iterations, enable_search, token_budget,
                  force_verdict, on_close):
    """Generator: streams (status, tokens, chat, verdict) updates while the debate runs."""
    chat: list[dict] = []
    iters = max(1, int(iterations or 1))

    if not topic or not position:
        yield "### ⚠️ Topic and position are required.", "", chat, ""
        return

    budget = int(token_budget) if token_budget else None
    meter = UsageMeter(hard_limit_tokens=budget)
    events: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    _active_stop["event"] = stop_event
    record = {"topic": topic, "meter": meter, "status": "running",
              "transcript": "", "detached": False}
    _runs.append(record)

    spec = DebateSpec(
        topic=topic,
        position=position,
        prosecution_provider=pros_provider,
        prosecution_model=_text(pros_model) or None,
        prosecution_instructions=_text(pros_instr),
        defense_provider=def_provider,
        defense_model=_text(def_model) or None,
        defense_instructions=_text(def_instr),
        moderator_provider=mod_provider,
        moderator_model=_text(mod_model) or None,
        moderator_instructions=_text(mod_instr),
        case_folder=_text(case_folder) or None,
        context_strategy=context_strategy or "auto",
        max_rounds=int(rounds),
        enable_search=bool(enable_search),
        max_total_tokens=budget,
        force_verdict=bool(force_verdict),
        verbose=False,
    )

    def on_status(stage, agent):
        events.put(("status", (stage, agent.name, agent.role.value, agent.provider.name)))

    def on_turn(speaker, role, content):
        events.put(("turn", (speaker, role, content)))

    def on_delta(speaker, role, fragment):
        events.put(("delta", (speaker, role, fragment)))

    def worker():
        outcomes = []
        try:
            for i in range(1, iters + 1):
                if stop_event.is_set():
                    break
                if iters > 1:
                    events.put(("iter", (i, iters)))
                outcome = run_debate(
                    spec, meter=meter,
                    on_turn=on_turn, on_status=on_status, on_delta=on_delta,
                    should_stop=stop_event.is_set,
                )
                outcomes.append(outcome)
                # A budget/cancel stop applies to the whole batch, not just
                # this iteration.
                if outcome.result.termination_reason in (
                    TerminationReason.TOKEN_BUDGET, TerminationReason.CANCELLED,
                ):
                    break
            winners = [o.result.winner for o in outcomes]
            tally = Counter(w or "undecided" for w in winners)
            record["status"] = "finished — " + (
                ", ".join(f"{w} {c}/{len(winners)}" for w, c in tally.most_common())
                if winners else "no runs"
            )
            record["transcript"] = str(outcomes[-1].transcript_path or "") if outcomes else ""
            events.put(("done", outcomes))
        except Exception as e:
            # Background-thread guard: report to the page, keep the traceback.
            logger.warning("Debate worker failed", exc_info=True)
            record["status"] = f"error: {str(e)[:120]}"
            events.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()

    status = "### 🏁 Starting debate…"
    stream_speaker = None  # (speaker, role) currently generating
    stream_text = ""

    def chat_view():
        """Committed turns plus the live-typing bubble, if any."""
        if not stream_speaker:
            return chat
        speaker, role = stream_speaker
        return chat + [{
            "role": "assistant",
            "content": f"**{ROLE_EMOJI.get(role, '💬')} {speaker} ({role})**\n\n{stream_text}▌",
        }]

    try:
        yield status, _fmt_tokens(meter, budget), chat, ""

        while True:
            # Wait for one event, then drain whatever else is queued so a
            # burst of stream deltas becomes a single UI update.
            try:
                batch = [events.get(timeout=0.4)]
            except queue.Empty:
                yield status, _fmt_tokens(meter, budget), gr.skip(), gr.skip()
                continue
            while True:
                try:
                    batch.append(events.get_nowait())
                except queue.Empty:
                    break

            finished = None
            for kind, payload in batch:
                if kind == "iter":
                    i, n = payload
                    chat = chat + [{
                        "role": "assistant",
                        "content": f"⚙️ — **Iteration {i} of {n}** —",
                    }]
                elif kind == "status":
                    stage, name, role, provider_name = payload
                    status = (f"### {ROLE_EMOJI.get(role, '💬')} Now running: **{name}** "
                              f"({stage})\n`{provider_name}`")
                    stream_speaker, stream_text = None, ""
                elif kind == "delta":
                    speaker, role, fragment = payload
                    if stream_speaker != (speaker, role):
                        stream_speaker, stream_text = (speaker, role), ""
                    stream_text += fragment
                elif kind == "turn":
                    speaker, role, content = payload
                    stream_speaker, stream_text = None, ""
                    chat = chat + [{
                        "role": "assistant",
                        "content": f"**{ROLE_EMOJI.get(role, '💬')} {speaker} ({role})**\n\n{content}",
                    }]
                elif kind in ("error", "done"):
                    finished = (kind, payload)

            if finished is None:
                yield status, _fmt_tokens(meter, budget), chat_view(), gr.skip()
                continue

            kind, payload = finished
            if kind == "error":
                yield f"### ❌ Error: {payload}", _fmt_tokens(meter, budget), chat, ""
                return

            outcomes = payload
            winners = [o.result.winner for o in outcomes]
            headline = verdict_headline(winners, position)
            detail_lines = [headline, ""]
            for i, o in enumerate(outcomes, 1):
                prefix = f"Iteration {i}: " if len(outcomes) > 1 else ""
                detail_lines.append(
                    f"- {prefix}**{(o.result.winner or 'undecided').upper()}** "
                    f"(`{o.result.termination_reason.value}`, "
                    f"{o.result.rounds_completed} round(s)) — `{o.transcript_path}`"
                )
            detail_lines.append(f"\n**Total cost:** {_cost_spans(meter)}")
            detail_lines.append(f"\n```\n{meter.summary()}\n```")
            # The headline doubles as the top status banner so the outcome is
            # unmissable in the final frame.
            yield headline, _fmt_tokens(meter, budget), chat, "\n".join(detail_lines)
            return
    except GeneratorExit:
        # Browser disconnected mid-run: honor the on-close preference.
        if record["status"] == "running":
            if str(on_close).startswith("Abort"):
                stop_event.set()
                record["status"] = "aborting (tab closed)"
            else:
                record["detached"] = True
        return
    finally:
        if _active_stop["event"] is stop_event:
            _active_stop["event"] = None


# -------------------------------------------------------- typed-in evidence

def save_text_evidence(entries_json, case_folder, section):
    """Persist typed entries as numbered evidence files in the case folder."""
    try:
        entries = [e for e in json.loads(entries_json or "[]") if e.strip()]
    except json.JSONDecodeError:
        entries = []
    if not entries:
        return "⚠️ Nothing to save — type at least one entry first."

    folder = Path(_text(case_folder) or "cases/manual_case")
    for sub in ("shared", "prosecution", "defense", "moderator"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    target = folder / section

    existing = len(list(target.glob("entry-*.txt")))
    saved = []
    for offset, text in enumerate(entries, start=existing + 1):
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()[:28]).strip("-") or "note"
        name = f"entry-{offset:02d}-{slug}.txt"
        (target / name).write_text(text + "\n")
        saved.append(name)
    return (f"✅ Saved {len(saved)} entr{'y' if len(saved) == 1 else 'ies'} to "
            f"`{target}`: {', '.join(saved)} — reload evidence to pick them up.")


def _text_evidence_panel(uid: str, case_folder):
    """Shared 'type evidence' editor: Enter = new entry, Shift+Enter = new line."""
    gr.Markdown(
        "**✍️ Or type evidence directly** — press **Enter** to start a new "
        "entry (separated by a faint line), **Shift+Enter** for a new line "
        "within the entry (bulleted lists etc.). Nothing is processed until "
        "you click Save."
    )
    gr.HTML(f"<div id='ec-editor-{uid}' class='ec-entry-host' "
            f"data-sync='ec-sync-{uid}'></div>")
    sync = gr.Textbox(elem_id=f"ec-sync-{uid}", elem_classes=["ec-hidden-sync"],
                      container=False)
    with gr.Row():
        section = gr.Dropdown(
            ["shared", "prosecution", "defense", "moderator"],
            value="shared", label="Save into", scale=1,
        )
        save_btn = gr.Button("💾 Save entries as evidence", size="sm", scale=1)
        clear_btn = gr.Button("🧹 Clear", size="sm", scale=0)
    status = gr.Markdown("")
    save_btn.click(save_text_evidence, inputs=[sync, case_folder, section],
                   outputs=[status])
    clear_btn.click(None, js=f"() => window.ecClearEditor('ec-editor-{uid}')")


# ---------------------------------------------------------------- ablation

_abl_stop = threading.Event()

_GRID_TRUTHY = {"x", "✓", "true", "1", "yes"}


def _grid_default(evidence_names: list) -> list:
    """Fresh manual-grid rows: 2 header rows + one row per evidence file."""
    n_cols = 3
    rows = [
        ["Scenario name", "baseline", "", ""][: n_cols + 1],
        ["# runs", "10", "", ""][: n_cols + 1],
    ]
    for name in evidence_names:
        rows.append([name] + [""] * n_cols)
    return rows


def parse_grid(rows: list) -> list:
    """Manual grid → scenarios. Row 0 = names, row 1 = # runs, then one row
    per evidence file; a truthy cell means that file is EXCLUDED there."""
    if not rows or len(rows) < 2:
        return []
    scenarios = []
    n_cols = len(rows[0])
    for col in range(1, n_cols):
        name = str(rows[0][col] or "").strip()
        if not name:
            continue
        try:
            runs = int(float(rows[1][col]))
        except (TypeError, ValueError):
            continue
        if runs <= 0:
            continue
        exclude = [
            str(row[0]).strip()
            for row in rows[2:]
            if str(row[col] if col < len(row) else "").strip().lower() in _GRID_TRUTHY
        ]
        scenarios.append(Scenario(name=name, runs=runs, exclude=exclude))
    return scenarios


def load_case_evidence(case_folder):
    """Populate the ablation controls from a case folder."""
    try:
        store = EvidenceStore.load(_text(case_folder))
        names = [n for n, _ in store.list_names()]
        sections = {n: s for n, s in store.list_names()}
        labels = [f"{n}  ({sections[n]})" for n in names]
        return (
            gr.CheckboxGroup(choices=names, value=names),
            gr.Dataframe(value=_grid_default(names)),
            names,
            f"✅ {len(names)} evidence files loaded: "
            + ", ".join(labels),
        )
    except Exception as e:
        # UI guard: shown to the user next to the Load button.
        logger.debug("Ablation evidence load failed", exc_info=True)
        return gr.skip(), gr.skip(), [], f"❌ {e}"


def ablation_estimate_md(provider, model, judge_provider, judge_model,
                         rounds, runs, ablate_selection, mode, grid, names):
    """Footprint for the whole campaign, funding-colored."""
    try:
        if mode == "Manual grid":
            scenarios = parse_grid(grid)
            total_runs = sum(s.runs for s in scenarios)
            n_scen = len(scenarios)
        else:
            n_scen = len(ablate_selection or []) + 1  # + baseline
            total_runs = int(runs or 0) * n_scen
        if total_runs <= 0:
            return "_No runs configured yet._"

        adv = resolve_model(provider, _text(model) or None, "advocate")
        judge = resolve_model(
            _text(judge_provider) or provider,
            _text(judge_model) or (_text(model) or None), "moderator",
        )
        fp = estimate_debate_footprint(adv, adv, judge, max_rounds=int(rounds or 1),
                                       iterations=total_runs)
        _, _, breakdown = estimate_run_cost(adv, adv, judge, max_rounds=int(rounds or 1))
        adv_cls = _fund_class(provider)
        judge_cls = _fund_class(_text(judge_provider) or provider)
        per_class: dict = {}
        for role_key, data in breakdown.items():
            cls = judge_cls if role_key == "moderator" else adv_cls
            per_class[cls] = per_class.get(cls, 0.0) + data["cost"] * total_runs
        parts = [
            _fund_span(cls, f"${per_class[cls] * 0.5:.2f}–${per_class[cls] * 1.5:.2f}")
            for cls in ("real", "credit", "included") if per_class.get(cls)
        ]
        cost_str = " · ".join(parts) if parts else "free (local models)"
        hours = fp["seconds"] / 3600
        time_str = f"~{hours:.1f} h" if hours >= 1 else f"~{fp['seconds'] / 60:.0f} min"
        return (f"**Campaign estimate:** {n_scen} scenarios · {total_runs} debates · "
                f"~{fp['tokens']:,} tokens · {cost_str} · {time_str} sequential "
                f"(divide by parallelism)" + _ESTIMATE_DISCLAIMER)
    except Exception as e:
        # UI guard: the estimate is advisory and reruns on every input change.
        logger.debug("Ablation estimate failed", exc_info=True)
        return f"_Estimate unavailable: {e}_"


def stop_ablation():
    _abl_stop.set()
    return "### ⏹️ Stopping after in-flight debates finish… (rerun to resume)"


def run_ablation_ui(case_folder, topic, position, provider, model,
                    judge_provider, judge_model, rounds, runs,
                    ablate_selection, protect_names, mode, grid,
                    budget_per_debate, campaign_budget, parallel, output_dir):
    """Generator: streams ablation progress into the tab."""
    if not _text(case_folder) or not _text(topic) or not _text(position):
        yield "### ⚠️ Case folder, topic, and position are required.", ""
        return

    _abl_stop.clear()
    scenarios = parse_grid(grid) if mode == "Manual grid" else []
    if mode == "Manual grid" and not scenarios:
        yield "### ⚠️ The grid has no runnable scenarios (need a name and # runs > 0).", ""
        return

    protect = [n for n in (protect_names or []) if n not in (ablate_selection or [])]
    config = AblationConfig(
        case_folder=_text(case_folder),
        topic=_text(topic),
        position=_text(position),
        provider=provider,
        model=_text(model) or None,
        judge_provider=_text(judge_provider) or None,
        judge_model=_text(judge_model) or None,
        rounds=int(rounds or 2),
        runs=int(runs or 10),
        protect=protect,
        scenarios=scenarios,
        max_total_tokens_per_debate=int(budget_per_debate) if budget_per_debate else None,
        max_campaign_tokens=int(campaign_budget) if campaign_budget else None,
        parallel=max(1, int(parallel or 1)),
        output_dir=_text(output_dir) or "./ablation/run",
    )

    events: queue.Queue = queue.Queue()
    done_box: dict = {}

    def worker():
        try:
            report = run_ablation(
                config,
                log=lambda s: None,
                on_progress=lambda n, total, rec: events.put(("run", (n, total, rec))),
                should_stop=_abl_stop.is_set,
            )
            done_box["report"] = report
            events.put(("done", report))
        except Exception as e:
            # Background-thread guard: report to the page, keep the traceback.
            logger.warning("Ablation worker failed", exc_info=True)
            events.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()
    status = "### 🧪 Ablation running…"
    cost = 0.0
    yield status, ""

    while True:
        try:
            kind, payload = events.get(timeout=1.0)
        except queue.Empty:
            yield gr.skip(), gr.skip()
            continue
        if kind == "run":
            n, total, rec = payload
            cost += rec["cost_usd"]
            margin = f" ({rec['margin']:+.0f})" if rec.get("margin") is not None else ""
            status = (f"### 🧪 Ablation: **{n}/{total}** debates — last: "
                      f"*{rec['scenario']}* → {rec['winner'] or 'undecided'}{margin} · "
                      f"spent {_fund_span(_fund_class(provider), f'${cost:.2f}')}")
            yield status, gr.skip()
        elif kind == "error":
            yield f"### ❌ Ablation error: {payload}", gr.skip()
            return
        elif kind == "done":
            report = payload
            yield ("### ✅ Ablation complete — "
                   f"{report['total_debates']} debates, ${report['total_cost_usd']}"),\
                  format_report(report)
            return


def _ablation_tab():
    gr.Markdown(
        "### 🧪 Evidence-salience ablation\n"
        "Run the case many times, then again with individual evidence removed, "
        "and measure how the verdict moves. **Auto** ablates every file "
        "(uncheck a file to protect it from ablation); **Manual grid** gives "
        "full control: one column per scenario, mark the files to *exclude*. "
        "Campaigns checkpoint to the output folder and resume if interrupted."
    )
    with gr.Row():
        with gr.Column(scale=1):
            case_folder = gr.Textbox(label="Case folder", placeholder="cases/example_case", elem_id="abl-case-folder")
            load_btn = gr.Button("📂 Load evidence", size="sm", elem_id="abl-load")
            load_status = gr.Markdown("")
            with gr.Accordion("✍️ Type evidence", open=False):
                _text_evidence_panel("ablation", case_folder)
            topic = gr.Textbox(label="Topic", elem_id="abl-topic")
            position = gr.Textbox(label="Prosecution position", elem_id="abl-position")
            with gr.Row():
                provider = gr.Dropdown(PROVIDERS, value=DEFAULT_PROVIDER,
                                       label="Advocates provider", scale=1)
                model = gr.Textbox(label="Model (blank = default)", scale=1)
            with gr.Row():
                judge_provider = gr.Dropdown([""] + PROVIDERS, value="",
                                             label="Judge provider (blank = same)", scale=1)
                judge_model = gr.Textbox(label="Judge model", scale=1)
            with gr.Row():
                rounds = gr.Slider(1, 6, value=2, step=1, label="Rounds", scale=1)
                runs = gr.Number(label="Runs per condition (auto mode)",
                                 value=20, precision=0, minimum=1, scale=1,
                                 elem_id="abl-runs")
            with gr.Row():
                budget = gr.Number(label="Token budget per debate",
                                   value=60_000, precision=0, scale=1)
                campaign_budget = gr.Number(
                    label="Campaign token budget (hard stop, all debates)",
                    value=2_000_000, precision=0, scale=1, elem_id="abl-campaign",
                )
                parallel = gr.Number(label="Parallel debates",
                                     value=4, precision=0, minimum=1, maximum=16, scale=1,
                                     elem_id="abl-parallel")
            output_dir = gr.Textbox(label="Output folder (checkpoint + report)",
                                    value="./ablation/run", elem_id="abl-output")

        with gr.Column(scale=2):
            mode = gr.Radio(["Auto (leave-one-out)", "Manual grid"],
                            value="Auto (leave-one-out)", label="Mode")
            ablate = gr.CheckboxGroup(
                [], label="Evidence to ablate (unchecked files are never removed)",
            )
            grid = gr.Dataframe(
                value=_grid_default([]),
                label="Manual grid — row 1: scenario name · row 2: # runs · "
                      "mark 'x' where a file is EXCLUDED",
                interactive=True,
                type="array",
            )
            estimate = gr.Markdown("_Load a case to see the campaign estimate._")
            with gr.Row():
                run_btn = gr.Button("🧪 Run ablation", variant="primary")
                stop_btn = gr.Button("⏹️ Stop")
            status = gr.Markdown("### 💤 Idle")
            results = gr.Markdown("")

    names_state = gr.State([])
    load_btn.click(
        load_case_evidence, inputs=[case_folder],
        outputs=[ablate, grid, names_state, load_status],
    )

    est_inputs = [provider, model, judge_provider, judge_model,
                  rounds, runs, ablate, mode, grid, names_state]
    for component in (provider, model, judge_provider, judge_model,
                      rounds, runs, ablate, mode):
        component.change(ablation_estimate_md, inputs=est_inputs, outputs=[estimate])

    run_btn.click(
        run_ablation_ui,
        inputs=[case_folder, topic, position, provider, model,
                judge_provider, judge_model, rounds, runs,
                ablate, names_state, mode, grid, budget, campaign_budget,
                parallel, output_dir],
        outputs=[status, results],
    )
    stop_btn.click(stop_ablation, outputs=[status])


# ------------------------------------------------------------------- layout

def _debate_tab():
    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Textbox(label="Topic", value="Tabs vs Spaces")
            position = gr.Textbox(
                label="Prosecution position",
                value="Tabs are superior to spaces",
            )

            with gr.Accordion("📁 Case evidence (optional)", open=False):
                gr.Markdown(
                    "Link a local case folder with `shared/` (visible to all sides) "
                    "plus proprietary `prosecution/`, `defense/`, and `moderator/` "
                    "subfolders (`.txt/.md/.json/.csv/.pdf`). Scaffold one with "
                    "`echochamber --init-case ./cases/my_case`."
                )
                case_folder = gr.Textbox(
                    label="Case folder path", placeholder="cases/example_case",
                )
                with gr.Row():
                    context_strategy = gr.Dropdown(
                        ["auto", "full", "summarize", "rag"], value="auto",
                        label="Context strategy", scale=1,
                    )
                    inspect_btn = gr.Button("🔍 Inspect evidence", size="sm", scale=1)
                evidence_info = gr.Markdown("")
                inspect_btn.click(inspect_evidence, inputs=[case_folder], outputs=[evidence_info])
                _text_evidence_panel("debate", case_folder)

            role_inputs = {}
            for role in ("prosecution", "defense", "moderator"):
                with gr.Group():
                    gr.Markdown(f"**{ROLE_EMOJI[role]} {role.capitalize()}**")
                    with gr.Row():
                        provider = gr.Dropdown(
                            PROVIDERS, value=DEFAULT_PROVIDER, label="Provider", scale=1,
                        )
                        model = gr.Textbox(
                            label="Model (blank = provider default)", scale=2,
                        )
                    instructions = gr.Textbox(
                        label="Custom system prompt additions (optional)",
                        placeholder="Appended to this agent's system prompt",
                        lines=2,
                    )
                role_inputs[role] = (provider, model, instructions)

            with gr.Accordion("🤖 AgentStable model recommendations", open=False):
                gr.Markdown(_recs_markdown())
                # Apply buttons set provider + model only (not instructions)
                rec_outputs = [w for triple in role_inputs.values() for w in triple[:2]]
                with gr.Row():
                    gr.Button("Apply #1 picks", size="sm").click(
                        lambda: _apply_recs(1), outputs=rec_outputs
                    )
                    gr.Button("Apply #2 picks", size="sm").click(
                        lambda: _apply_recs(2), outputs=rec_outputs
                    )

            with gr.Row():
                rounds = gr.Slider(1, 8, value=2, step=1, label="Max rounds", scale=2)
                iterations = gr.Number(
                    label="Iterations (repeat the debate)",
                    value=1, precision=0, minimum=1, maximum=20, scale=1,
                )
            enable_search = gr.Checkbox(label="Enable web search", value=False)

            estimate = gr.Markdown(estimate_md(
                DEFAULT_PROVIDER, "", DEFAULT_PROVIDER, "", DEFAULT_PROVIDER, "",
                2, 1, 200_000,
            ))
            token_budget = gr.Number(
                label="Hard token budget (total across all agents and iterations)",
                value=200_000, precision=0,
            )
            force_verdict = gr.Checkbox(
                label="Force a verdict before the budget runs out "
                      "(reserves ~10,000 tokens for the ruling)",
                value=True,
            )
            estimate_inputs = [
                role_inputs["prosecution"][0], role_inputs["prosecution"][1],
                role_inputs["defense"][0], role_inputs["defense"][1],
                role_inputs["moderator"][0], role_inputs["moderator"][1],
                rounds, iterations, token_budget,
            ]
            for component in estimate_inputs:
                component.change(estimate_md, inputs=estimate_inputs, outputs=[estimate])
            on_close = gr.Radio(
                ["Abort the debate", "Continue in background"],
                value="Abort the debate",
                label="If I close this tab mid-debate",
            )

            with gr.Row():
                run_btn = gr.Button("▶️ Run debate", variant="primary")
                stop_btn = gr.Button("⏹️ Stop")

            background = gr.Markdown(background_md())
            gr.Timer(5).tick(background_md, outputs=[background])

        with gr.Column(scale=2):
            status = gr.Markdown("### 💤 Idle")
            tokens = gr.Markdown("")
            chatbot = gr.Chatbot(label="Proceedings", height=520)
            verdict = gr.Markdown("")

    inputs = [topic, position]
    for role in ("prosecution", "defense", "moderator"):
        inputs.extend(role_inputs[role])
    inputs.extend([case_folder, context_strategy, rounds, iterations,
                   enable_search, token_budget, force_verdict, on_close])

    # The running-flag is toggled by dedicated js-only listeners so the main
    # event's input payload is never transformed client-side.
    run_btn.click(None, js="() => { window.__ec_running = true; }")
    run_event = run_btn.click(
        run_debate_ui,
        inputs=inputs,
        outputs=[status, tokens, chatbot, verdict],
    )
    run_event.then(None, js="() => { window.__ec_running = false; }")
    stop_btn.click(stop_debate, outputs=[status])


def _setup_tab():
    gr.Markdown(
        "### 🔑 Provider API keys\n"
        "Paste a key, **Test** it, then **Save**. Keys go into a local `.env` file "
        "readable only by your user account, and take effect immediately. "
        "Only fill the providers you plan to use — LM Studio (local) needs no key.\n\n"
        "Tip: create keys with a low monthly spend limit in each provider's console."
    )
    overview = gr.Markdown(key_overview_md())

    boxes = {}
    for provider in ("anthropic", "openai", "together", "gemini"):
        with gr.Row():
            box = gr.Textbox(
                label=f"{provider} — {PROVIDER_ENV_KEYS[provider]}",
                type="password", scale=3,
            )
            test_btn = gr.Button("Test", size="sm", scale=1)
        result = gr.Markdown("")
        test_btn.click(
            lambda key, p=provider: test_provider_key(p, key),
            inputs=[box], outputs=[result],
        )
        boxes[provider] = box

    with gr.Row():
        lms_btn = gr.Button("Test LM Studio (local, no key)", size="sm")
        lms_result = gr.Markdown("")
        lms_btn.click(lambda: test_provider_key("lmstudio", ""), outputs=[lms_result])

    save_btn = gr.Button("💾 Save keys", variant="primary")
    save_status = gr.Markdown("")
    save_btn.click(
        save_keys,
        inputs=[boxes["anthropic"], boxes["openai"], boxes["together"], boxes["gemini"]],
        outputs=[overview, save_status],
    )


def build_app() -> "gr.Blocks":
    if gr is None:  # pragma: no cover
        raise SystemExit(_GRADIO_MISSING)
    with gr.Blocks(title="EchoChamber", head=_HEAD_JS) as app:
        gr.Markdown("# ⚖️ EchoChamber — Multi-LLM Courtroom Debate")
        with gr.Tab("Debate"):
            _debate_tab()
        with gr.Tab("Ablation"):
            _ablation_tab()
        with gr.Tab("Setup"):
            _setup_tab()
    return app


def main():
    # Container/proxy friendly: honor $PORT (Cloud Run), bind address, and a
    # root path for serving behind a reverse-proxy prefix like /echochamber.
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get(
        "ECHOCHAMBER_HOST",
        "0.0.0.0" if "PORT" in os.environ else "127.0.0.1",
    )
    root_path = os.environ.get("ECHOCHAMBER_ROOT_PATH", "") or None
    build_app().launch(
        theme=gr.themes.Soft(),
        server_name=host,
        server_port=port,
        root_path=root_path,
    )


if __name__ == "__main__":
    main()
