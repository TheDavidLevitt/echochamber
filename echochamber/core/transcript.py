"""Transcript logging for court sessions."""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..providers import Message


@dataclass
class RunConfig:
    """Configuration used for a debate run (for transcript metadata)."""
    topic: str = ""
    position: str = ""
    prosecution_provider: str = ""
    prosecution_model: str = ""
    defense_provider: str = ""
    defense_model: str = ""
    moderator_provider: str = ""
    moderator_model: str = ""
    max_rounds: int = 0
    allow_concession: bool = False
    allow_conviction: bool = False
    context_strategy: str = ""
    enable_search: bool = False
    case_folder: str = ""


@dataclass
class TranscriptEntry:
    """A single entry in the transcript."""
    speaker: str
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    round_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Transcript:
    """
    Full transcript of a court session.

    Tracks all statements, decisions, and metadata.
    """
    case_topic: str
    entries: list[TranscriptEntry] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    outcome: Optional[str] = None
    winner: Optional[str] = None
    total_rounds: int = 0
    run_config: Optional[RunConfig] = None

    def add_entry(
        self,
        speaker: str,
        role: str,
        content: str,
        round_number: Optional[int] = None,
        **metadata,
    ) -> None:
        """Add a new entry to the transcript."""
        entry = TranscriptEntry(
            speaker=speaker,
            role=role,
            content=content,
            round_number=round_number,
            metadata=metadata,
        )
        self.entries.append(entry)

    def get_conversation_history(self) -> list[Message]:
        """
        Convert transcript to Message list for LLM context.

        Returns messages in alternating user/assistant format,
        with speaker names preserved in content.
        """
        messages = []
        for entry in self.entries:
            # Format as user messages with speaker attribution
            # This keeps the conversation visible to all parties
            content = f"[{entry.speaker} ({entry.role})]:\n{entry.content}"
            messages.append(Message(
                role="user",
                content=content,
                name=entry.speaker,
            ))
        return messages

    def finalize(
        self,
        outcome: str,
        winner: Optional[str] = None,
    ) -> None:
        """Mark the transcript as complete."""
        self.ended_at = datetime.now().isoformat()
        self.outcome = outcome
        self.winner = winner

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "case_topic": self.case_topic,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": self.outcome,
            "winner": self.winner,
            "total_rounds": self.total_rounds,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: str | Path) -> None:
        """Save transcript to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        """Load transcript from JSON file."""
        with open(path) as f:
            data = json.load(f)

        transcript = cls(
            case_topic=data["case_topic"],
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at"),
            outcome=data.get("outcome"),
            winner=data.get("winner"),
            total_rounds=data.get("total_rounds", 0),
        )

        for entry_data in data.get("entries", []):
            transcript.entries.append(TranscriptEntry(**entry_data))

        return transcript

    def __str__(self) -> str:
        """Human-readable transcript."""
        lines = [
            "=== COURT TRANSCRIPT ===",
            f"Case: {self.case_topic}",
            f"Started: {self.started_at}",
            "",
        ]

        current_round = None
        for entry in self.entries:
            if entry.round_number and entry.round_number != current_round:
                current_round = entry.round_number
                lines.append(f"\n--- Round {current_round} ---\n")

            lines.append(f"[{entry.speaker} ({entry.role})]:")
            lines.append(entry.content)
            lines.append("")

        if self.outcome:
            lines.append(f"\n=== OUTCOME: {self.outcome} ===")
            if self.winner:
                lines.append(f"Winner: {self.winner}")

        return "\n".join(lines)

    def generate_run_id(self, output_dir: Path) -> str:
        """
        Generate a unique run ID in format: DATE-LLMs-ROUNDS-ITERATION

        Example: 20260202-lmstudio-lmstudio-lmstudio-R5-001

        Args:
            output_dir: Directory where transcripts are saved (to count iterations)

        Returns:
            Unique run ID string
        """
        # Date component
        start_dt = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
        date_str = start_dt.strftime("%Y%m%d")

        # LLM providers component (shortened)
        if self.run_config:
            providers = [
                self._shorten_provider(self.run_config.prosecution_provider),
                self._shorten_provider(self.run_config.defense_provider),
                self._shorten_provider(self.run_config.moderator_provider),
            ]
            llm_str = "-".join(providers)
        else:
            llm_str = "unknown"

        # Rounds component
        rounds_str = f"R{self.total_rounds}"

        # Iteration component - count existing files with same prefix
        base_prefix = f"{date_str}-{llm_str}-{rounds_str}"
        iteration = 1

        if output_dir.exists():
            existing = list(output_dir.glob(f"{base_prefix}-*.md"))
            if existing:
                # Extract iteration numbers and find max
                for f in existing:
                    match = re.search(r'-(\d{3})\.md$', f.name)
                    if match:
                        iteration = max(iteration, int(match.group(1)) + 1)

        return f"{base_prefix}-{iteration:03d}"

    def _shorten_provider(self, provider: str) -> str:
        """Shorten provider name for run ID."""
        shortcuts = {
            "anthropic": "ant",
            "openai": "oai",
            "together": "tog",
            "lmstudio": "lms",
            "gemini": "gem",
        }
        return shortcuts.get(provider.lower(), provider[:3])

    def to_markdown(self, run_id: Optional[str] = None) -> str:
        """
        Generate a markdown transcript with summary and full transcript.

        Args:
            run_id: Optional run ID (generated if not provided)

        Returns:
            Markdown formatted string
        """
        lines = []

        # Header
        lines.append("# EchoChamber Debate Transcript")
        lines.append("")

        # Run ID
        if run_id:
            lines.append(f"**Run ID:** `{run_id}`")
            lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Topic:** {self.case_topic}")
        lines.append("")

        if self.run_config and self.run_config.position:
            lines.append(f"**Prosecution Position:** {self.run_config.position}")
            lines.append("")

        # Timestamp
        start_dt = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
        lines.append(f"**Started:** {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.ended_at:
            end_dt = datetime.fromisoformat(self.ended_at.replace('Z', '+00:00'))
            lines.append(f"**Ended:** {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = end_dt - start_dt
            lines.append(f"**Duration:** {duration}")
        lines.append("")

        # Verdict
        lines.append("### Verdict")
        lines.append("")
        if self.winner:
            winner_display = self.winner.upper()
            lines.append(f"**Winner:** {winner_display}")
        else:
            lines.append("**Winner:** UNDECIDED")
        lines.append(f"**Rounds Completed:** {self.total_rounds}")
        if self.outcome:
            lines.append(f"**Termination:** {self.outcome}")
        lines.append("")

        # Run parameters
        if self.run_config:
            lines.append("### Run Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Prosecution | {self.run_config.prosecution_provider}/{self.run_config.prosecution_model} |")
            lines.append(f"| Defense | {self.run_config.defense_provider}/{self.run_config.defense_model} |")
            lines.append(f"| Moderator | {self.run_config.moderator_provider}/{self.run_config.moderator_model} |")
            lines.append(f"| Max Rounds | {self.run_config.max_rounds} |")
            lines.append(f"| Allow Concession | {self.run_config.allow_concession} |")
            lines.append(f"| Allow Conviction | {self.run_config.allow_conviction} |")
            lines.append(f"| Context Strategy | {self.run_config.context_strategy} |")
            lines.append(f"| Web Search | {self.run_config.enable_search} |")
            if self.run_config.case_folder:
                lines.append(f"| Case Folder | {self.run_config.case_folder} |")
            lines.append("")

        # Horizontal rule before transcript
        lines.append("---")
        lines.append("")

        # Full transcript
        lines.append("## Full Transcript")
        lines.append("")

        current_round = None
        for entry in self.entries:
            # Round header
            if entry.round_number and entry.round_number != current_round:
                current_round = entry.round_number
                lines.append(f"### Round {current_round}")
                lines.append("")
            elif current_round is None and entry.round_number is None:
                # Opening statements (before rounds)
                if not any("Opening" in line for line in lines[-5:]):
                    lines.append("### Opening")
                    lines.append("")

            # Speaker header
            role_emoji = {
                "prosecution": "⚖️",
                "defense": "🛡️",
                "moderator": "👨‍⚖️",
            }.get(entry.role.lower(), "💬")

            lines.append(f"#### {role_emoji} {entry.speaker} ({entry.role.title()})")
            lines.append("")
            lines.append(entry.content)
            lines.append("")

        # Final outcome
        lines.append("---")
        lines.append("")
        lines.append("## Final Outcome")
        lines.append("")
        if self.winner:
            lines.append(f"**{self.winner.upper()} WINS**")
        else:
            lines.append("**UNDECIDED**")
        lines.append("")

        return "\n".join(lines)

    def save_markdown(self, output_dir: str | Path, run_id: Optional[str] = None) -> Path:
        """
        Save transcript as a markdown file.

        Args:
            output_dir: Directory to save the transcript
            run_id: Optional run ID (generated if not provided)

        Returns:
            Path to the saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID if not provided
        if not run_id:
            run_id = self.generate_run_id(output_dir)

        # Generate markdown content
        content = self.to_markdown(run_id)

        # Save file
        file_path = output_dir / f"{run_id}.md"
        file_path.write_text(content, encoding="utf-8")

        return file_path
