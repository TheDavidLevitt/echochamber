"""Evidence management for court sessions.

Evidence can be:
- Shared: Visible to all parties from the start
- Private: Only visible to the owning party, can be "introduced" to make it shared

Folder structure:
    cases/
    └── my_case/
        ├── shared/           # All parties see this
        │   └── background.md
        ├── prosecution/      # Only prosecution sees until introduced
        │   └── witness_statement.txt
        ├── defense/          # Only defense sees until introduced
        │   └── alibi_evidence.pdf
        └── moderator/        # Only moderator sees (for guidance)
            └── scoring_rubric.md

Markdown files can have special sections:
- ## Instructions: Added to agent's system prompt (not shown as evidence)
- ## Context: Shown as private context/facts for the agent
- Other sections: Treated as regular evidence content
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent import Role

logger = logging.getLogger(__name__)


def parse_markdown_sections(content: str) -> dict[str, str]:
    """
    Parse a markdown file into sections based on ## headers.

    Returns a dict with keys like 'instructions', 'context', and 'content'.
    'content' contains everything that's not a special section.
    """
    sections = {
        'instructions': '',
        'context': '',
        'content': '',
    }

    # Split by ## headers
    pattern = r'^##\s+(.+)$'
    parts = re.split(pattern, content, flags=re.MULTILINE)

    # parts[0] is content before any ## header
    current_content_parts = [parts[0].strip()] if parts[0].strip() else []

    # Process header/content pairs
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip().lower()
        body = parts[i + 1].strip()

        if header == 'instructions':
            sections['instructions'] = body
        elif header == 'context':
            sections['context'] = body
        else:
            # Regular section - include header and body in content
            current_content_parts.append(f"## {parts[i].strip()}\n{body}")

        i += 2

    sections['content'] = '\n\n'.join(current_content_parts)
    return sections


# Supported file extensions
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def extract_pdf_text(file_path: Path) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "PDF support requires pymupdf. Install with: pip install pymupdf"
        )

    text_parts = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(f"[Page {page_num}]\n{page_text}")

    doc.close()

    return "\n\n".join(text_parts)


def get_pdf_page_count(file_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        import fitz
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
    except (ImportError, OSError, RuntimeError, ValueError):
        # pymupdf missing, or the file is unreadable/not a PDF (pymupdf's
        # FileDataError is a RuntimeError)
        logger.debug("Could not count pages in %s", file_path, exc_info=True)
        return 0


@dataclass
class Evidence:
    """A single piece of evidence."""
    name: str
    content: str
    source_path: str
    owner: Optional[str] = None  # None = shared, otherwise role name
    introduced: bool = False  # Private evidence that has been shared
    page_count: int = 0  # For PDFs
    char_count: int = 0  # Character count for context estimation
    instructions: str = ""  # Instructions to add to agent system prompt
    context: str = ""  # Private context/facts for the agent

    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.content)

    def __str__(self) -> str:
        status = "shared" if self.owner is None else f"private ({self.owner})"
        if self.introduced:
            status = "introduced"
        pages = f", {self.page_count} pages" if self.page_count else ""
        has_instructions = " +instr" if self.instructions else ""
        return f"[{self.name}] ({status}{pages}{has_instructions})"


@dataclass
class EvidenceStore:
    """
    Manages evidence from a case folder.

    Evidence is organized by visibility:
    - shared/: Visible to all parties
    - prosecution/: Only prosecution can see (until introduced)
    - defense/: Only defense can see (until introduced)
    - moderator/: Only moderator can see
    """
    case_path: Path
    shared: list[Evidence] = field(default_factory=list)
    prosecution: list[Evidence] = field(default_factory=list)
    defense: list[Evidence] = field(default_factory=list)
    moderator: list[Evidence] = field(default_factory=list)
    introduced: list[Evidence] = field(default_factory=list)  # Previously private, now shared

    @classmethod
    def load(
        cls,
        case_path: str | Path,
        exclude: Optional[list] = None,
    ) -> "EvidenceStore":
        """
        Load evidence from a case folder.

        Args:
            case_path: Path to the case folder
            exclude: Evidence file names to drop after loading (used by
                ablation studies to run "the same case minus this document")

        Returns:
            Populated EvidenceStore
        """
        case_path = Path(case_path)
        if not case_path.exists():
            raise ValueError(f"Case folder not found: {case_path}")

        store = cls(case_path=case_path)

        # Load each category
        store.shared = store._load_folder(case_path / "shared", owner=None)
        store.prosecution = store._load_folder(case_path / "prosecution", owner="prosecution")
        store.defense = store._load_folder(case_path / "defense", owner="defense")
        store.moderator = store._load_folder(case_path / "moderator", owner="moderator")

        if exclude:
            dropped = set(exclude)
            store.shared = [e for e in store.shared if e.name not in dropped]
            store.prosecution = [e for e in store.prosecution if e.name not in dropped]
            store.defense = [e for e in store.defense if e.name not in dropped]
            store.moderator = [e for e in store.moderator if e.name not in dropped]

        return store

    def list_names(self) -> list[tuple[str, str]]:
        """All evidence as (file_name, section) pairs, section order preserved."""
        out = []
        for section, items in (
            ("shared", self.shared), ("prosecution", self.prosecution),
            ("defense", self.defense), ("moderator", self.moderator),
        ):
            out.extend((e.name, section) for e in items)
        return out

    def _load_folder(self, folder: Path, owner: Optional[str]) -> list[Evidence]:
        """Load all supported files from a folder."""
        evidence_list = []

        if not folder.exists():
            return evidence_list

        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            try:
                if suffix in SUPPORTED_PDF_EXTENSIONS:
                    content = extract_pdf_text(file_path)
                    page_count = get_pdf_page_count(file_path)
                    evidence_list.append(Evidence(
                        name=file_path.name,
                        content=content,
                        source_path=str(file_path),
                        owner=owner,
                        page_count=page_count,
                    ))
                elif suffix == ".md":
                    # Parse markdown for special sections
                    raw_content = file_path.read_text(encoding="utf-8")
                    sections = parse_markdown_sections(raw_content)
                    evidence_list.append(Evidence(
                        name=file_path.name,
                        content=sections['content'],
                        source_path=str(file_path),
                        owner=owner,
                        instructions=sections['instructions'],
                        context=sections['context'],
                    ))
                else:
                    content = file_path.read_text(encoding="utf-8")
                    evidence_list.append(Evidence(
                        name=file_path.name,
                        content=content,
                        source_path=str(file_path),
                        owner=owner,
                    ))
            except Exception as e:
                # Per-file guard: one bad file (encoding, corrupt PDF, missing
                # extra, ...) is skipped instead of failing the whole case.
                print(f"Warning: Could not load {file_path}: {e}")
                logger.debug("Evidence load failed for %s", file_path, exc_info=True)

        return evidence_list

    def introduce_evidence(self, name: str, from_role: str) -> bool:
        """
        Introduce private evidence, making it visible to all parties.

        Args:
            name: Name of the evidence file
            from_role: Role introducing the evidence ("prosecution" or "defense")

        Returns:
            True if evidence was found and introduced
        """
        source_list = self.prosecution if from_role == "prosecution" else self.defense

        for i, evidence in enumerate(source_list):
            if evidence.name == name:
                evidence.introduced = True
                self.introduced.append(evidence)
                return True

        return False

    def get_context_for_role(self, role: Role) -> str:
        """
        Get all evidence visible to a specific role as formatted text.

        Args:
            role: The role requesting evidence

        Returns:
            Formatted string with all visible evidence
        """
        visible = []

        # Shared evidence is always visible
        visible.extend(self.shared)

        # Introduced evidence is visible to all
        visible.extend(self.introduced)

        # Role-specific private evidence
        if role == Role.PROSECUTION:
            # Prosecution sees their private evidence (excluding already introduced)
            visible.extend([e for e in self.prosecution if not e.introduced])
        elif role == Role.DEFENSE:
            visible.extend([e for e in self.defense if not e.introduced])
        elif role == Role.MODERATOR:
            # Moderator sees their own guidance docs
            visible.extend(self.moderator)
            # Moderator can optionally see all evidence (for fair evaluation)
            # Uncomment below to enable:
            # visible.extend([e for e in self.prosecution if not e.introduced])
            # visible.extend([e for e in self.defense if not e.introduced])

        if not visible:
            return ""

        sections = ["=== AVAILABLE EVIDENCE ===\n"]
        for evidence in visible:
            status = ""
            if evidence.owner and not evidence.introduced:
                status = " [PRIVATE - only you can see this]"
            elif evidence.introduced:
                status = " [INTRODUCED]"

            sections.append(f"--- {evidence.name}{status} ---")
            sections.append(evidence.content)
            sections.append("")

        return "\n".join(sections)

    def get_shared_context(self) -> str:
        """Get only the shared/introduced evidence (visible to all)."""
        visible = self.shared + self.introduced

        if not visible:
            return ""

        sections = ["=== CASE EVIDENCE ===\n"]
        for evidence in visible:
            sections.append(f"--- {evidence.name} ---")
            sections.append(evidence.content)
            sections.append("")

        return "\n".join(sections)

    def get_instructions_for_role(self, role: Role) -> str:
        """
        Get instructions from evidence files for a specific role.

        Instructions come from ## Instructions sections in .md files
        and are intended to be added to the agent's system prompt.
        """
        instructions_parts = []

        # Get instructions from role-specific evidence
        if role == Role.PROSECUTION:
            for e in self.prosecution:
                if e.instructions:
                    instructions_parts.append(e.instructions)
        elif role == Role.DEFENSE:
            for e in self.defense:
                if e.instructions:
                    instructions_parts.append(e.instructions)
        elif role == Role.MODERATOR:
            for e in self.moderator:
                if e.instructions:
                    instructions_parts.append(e.instructions)

        return "\n\n".join(instructions_parts)

    def get_private_context_for_role(self, role: Role) -> str:
        """
        Get private context from evidence files for a specific role.

        Context comes from ## Context sections in .md files
        and provides private facts/background for the agent.
        """
        context_parts = []

        if role == Role.PROSECUTION:
            for e in self.prosecution:
                if e.context:
                    context_parts.append(f"[From {e.name}]\n{e.context}")
        elif role == Role.DEFENSE:
            for e in self.defense:
                if e.context:
                    context_parts.append(f"[From {e.name}]\n{e.context}")
        elif role == Role.MODERATOR:
            for e in self.moderator:
                if e.context:
                    context_parts.append(f"[From {e.name}]\n{e.context}")

        if not context_parts:
            return ""

        return "=== PRIVATE CONTEXT ===\n" + "\n\n".join(context_parts)

    def total_chars(self) -> int:
        """Get total character count across all evidence."""
        all_evidence = self.shared + self.prosecution + self.defense + self.moderator
        return sum(e.char_count for e in all_evidence)

    def total_pages(self) -> int:
        """Get total PDF page count across all evidence."""
        all_evidence = self.shared + self.prosecution + self.defense + self.moderator
        return sum(e.page_count for e in all_evidence)

    def summary(self) -> str:
        """Get a summary of loaded evidence."""
        all_evidence = self.shared + self.prosecution + self.defense + self.moderator
        total_chars = sum(e.char_count for e in all_evidence)
        total_pages = sum(e.page_count for e in all_evidence)

        # Estimate tokens (~4 chars per token for English)
        est_tokens = total_chars // 4

        lines = [
            f"Evidence loaded from: {self.case_path}",
            f"  Shared: {len(self.shared)} files",
            f"  Prosecution (private): {len(self.prosecution)} files",
            f"  Defense (private): {len(self.defense)} files",
            f"  Moderator: {len(self.moderator)} files",
            f"  Total: {total_chars:,} chars (~{est_tokens:,} tokens)",
        ]
        if total_pages:
            lines.append(f"  PDF pages: {total_pages}")

        return "\n".join(lines)


def create_case_folder(path: str | Path, topic: str = "") -> Path:
    """
    Create a new case folder with the standard structure.

    Args:
        path: Where to create the case folder
        topic: Optional topic to include in a README

    Returns:
        Path to the created case folder
    """
    case_path = Path(path)
    case_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (case_path / "shared").mkdir(exist_ok=True)
    (case_path / "prosecution").mkdir(exist_ok=True)
    (case_path / "defense").mkdir(exist_ok=True)
    (case_path / "moderator").mkdir(exist_ok=True)

    # Create a README
    readme_content = f"""# Case: {topic or 'Untitled'}

## Folder Structure

- `shared/` - Evidence visible to all parties from the start
- `prosecution/` - Private evidence for prosecution (can be introduced during debate)
- `defense/` - Private evidence for defense (can be introduced during debate)
- `moderator/` - Guidance documents for the moderator only

## Supported File Types

- `.txt` - Plain text
- `.md` - Markdown
- `.json` - JSON data
- `.csv` - Comma-separated values
- `.log` - Log files
- `.pdf` - PDF documents

## Usage

Place your evidence files in the appropriate folders, then run:

```bash
python -m echochamber.cli --topic "Your topic" --position "Position" --case-folder {case_path}
```
"""
    (case_path / "README.md").write_text(readme_content)

    return case_path
