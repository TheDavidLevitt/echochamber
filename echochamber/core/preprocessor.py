"""Document preprocessing for large evidence sets.

Supports three strategies:
1. FULL - Load complete documents (for small evidence sets or large context models)
2. SUMMARIZE - Pre-summarize documents to fit context limits
3. RAG - Use retrieval-augmented generation with embeddings

The preprocessor can use the same LLM providers as the debate agents.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from .evidence import Evidence, EvidenceStore

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategy for handling document context."""
    AUTO = "auto"           # Automatically select based on document size
    FULL = "full"           # Use complete documents
    SUMMARIZE = "summarize" # Pre-summarize to fit context
    RAG = "rag"             # Retrieve relevant chunks per query


def auto_select_strategy(
    total_chars: int,
    max_context_tokens: int,
    num_documents: int,
) -> ContextStrategy:
    """
    Automatically select the best context strategy based on document size.

    Args:
        total_chars: Total characters across all documents
        max_context_tokens: Maximum tokens the model can handle
        num_documents: Number of documents

    Returns:
        Recommended ContextStrategy
    """
    # Estimate tokens (roughly 4 chars per token for English)
    estimated_tokens = total_chars // 4

    # Reserve ~30% of context for conversation history and prompts
    available_tokens = int(max_context_tokens * 0.7)

    if estimated_tokens <= available_tokens:
        # Documents fit in context - use full
        return ContextStrategy.FULL
    elif estimated_tokens <= available_tokens * 3:
        # Moderate overflow - summarization can help
        # Summarization typically achieves 3-5x compression
        return ContextStrategy.SUMMARIZE
    else:
        # Large overflow - need RAG
        return ContextStrategy.RAG


@dataclass
class PreprocessorConfig:
    """Configuration for document preprocessing."""
    strategy: ContextStrategy = ContextStrategy.FULL
    max_tokens: int = 100_000      # Target token limit for context
    chunk_size: int = 1000         # Characters per chunk for RAG
    chunk_overlap: int = 200       # Overlap between chunks
    top_k: int = 10                # Number of chunks to retrieve for RAG
    summary_max_tokens: int = 500  # Max tokens per document summary
    cache_dir: Optional[Path] = None  # Cache summaries/embeddings


@dataclass
class ProcessedEvidence:
    """Evidence after preprocessing."""
    original: Evidence
    processed_content: str  # May be summary or full content
    chunks: list[str] = field(default_factory=list)  # For RAG
    chunk_embeddings: list[list[float]] = field(default_factory=list)
    summary: Optional[str] = None
    token_estimate: int = 0

    def __post_init__(self):
        self.token_estimate = len(self.processed_content) // 4


class DocumentPreprocessor:
    """
    Preprocesses documents based on configured strategy.

    For summarization, requires an LLM provider.
    For RAG, chunks go into an in-memory ChromaDB collection using Chroma's
    default embedding function (all-MiniLM-L6-v2 run locally via ONNX
    Runtime, downloaded on first use); sentence-transformers is not used.
    """

    def __init__(
        self,
        config: PreprocessorConfig,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
            summarizer: Optional function to summarize text (for SUMMARIZE strategy)
                       Should accept text and return summary
        """
        self.config = config
        self.summarizer = summarizer
        self._embedding_model = None
        self._vector_store = None

        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def process_evidence_store(self, store: EvidenceStore) -> "ProcessedEvidenceStore":
        """
        Process all evidence in a store according to the configured strategy.

        Args:
            store: The evidence store to process

        Returns:
            ProcessedEvidenceStore with processed evidence
        """
        processed = ProcessedEvidenceStore(
            original=store,
            config=self.config,
        )

        all_evidence = store.shared + store.prosecution + store.defense + store.moderator

        for evidence in all_evidence:
            proc = self._process_single(evidence)

            # Categorize by owner
            if evidence.owner is None:
                processed.shared.append(proc)
            elif evidence.owner == "prosecution":
                processed.prosecution.append(proc)
            elif evidence.owner == "defense":
                processed.defense.append(proc)
            elif evidence.owner == "moderator":
                processed.moderator.append(proc)

        # Initialize RAG index if needed
        if self.config.strategy == ContextStrategy.RAG:
            processed._build_rag_index()

        return processed

    def _process_single(self, evidence: Evidence) -> ProcessedEvidence:
        """Process a single piece of evidence."""
        if self.config.strategy == ContextStrategy.FULL:
            return ProcessedEvidence(
                original=evidence,
                processed_content=evidence.content,
            )

        elif self.config.strategy == ContextStrategy.SUMMARIZE:
            summary = self._summarize(evidence)
            return ProcessedEvidence(
                original=evidence,
                processed_content=summary,
                summary=summary,
            )

        elif self.config.strategy == ContextStrategy.RAG:
            chunks = self._chunk_document(evidence.content)
            return ProcessedEvidence(
                original=evidence,
                processed_content=f"[Document: {evidence.name} - {len(chunks)} chunks available for retrieval]",
                chunks=chunks,
            )

        return ProcessedEvidence(original=evidence, processed_content=evidence.content)

    def _summarize(self, evidence: Evidence) -> str:
        """Summarize a document."""
        # Check cache first
        cache_key = self._cache_key(evidence, "summary")
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        if not self.summarizer:
            # Fallback: truncate to first N characters
            max_chars = self.config.summary_max_tokens * 4
            if len(evidence.content) <= max_chars:
                return evidence.content
            return evidence.content[:max_chars] + f"\n\n[... truncated, {len(evidence.content) - max_chars} chars omitted ...]"

        # Use provided summarizer
        prompt = f"""Summarize the following document concisely, preserving key facts, arguments, and evidence that would be relevant in a legal/debate context.

Document: {evidence.name}

Content:
{evidence.content[:50000]}  # Limit input to avoid issues

Provide a summary of no more than {self.config.summary_max_tokens * 4} characters."""

        summary = self.summarizer(prompt)

        # Cache the result
        self._save_cache(cache_key, summary)

        return summary

    def _chunk_document(self, content: str) -> list[str]:
        """Split document into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.config.chunk_size
            chunk = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind('. ')
                if last_period > self.config.chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.config.chunk_overlap

        return chunks

    def _cache_key(self, evidence: Evidence, prefix: str) -> str:
        """Generate a cache key for evidence."""
        content_hash = hashlib.md5(evidence.content.encode()).hexdigest()[:12]
        return f"{prefix}_{evidence.name}_{content_hash}"

    def _load_cache(self, key: str) -> Optional[str]:
        """Load from cache if available."""
        if not self.config.cache_dir:
            return None
        cache_file = self.config.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return data.get("content")
            except (OSError, ValueError, AttributeError) as e:
                # Unreadable, corrupt (ValueError covers JSON and Unicode
                # decode errors) or not a JSON object: treat as a cache miss.
                logger.warning("Ignoring unreadable cache file %s: %s", cache_file, e)
        return None

    def _save_cache(self, key: str, content: str) -> None:
        """Save to cache."""
        if not self.config.cache_dir:
            return
        cache_file = self.config.cache_dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps({"content": content}))
        except OSError as e:
            # Caching is an optimization; a failed write only costs a recompute.
            logger.warning("Could not write cache file %s: %s", cache_file, e)


@dataclass
class ProcessedEvidenceStore:
    """Evidence store after preprocessing."""
    original: EvidenceStore
    config: PreprocessorConfig
    shared: list[ProcessedEvidence] = field(default_factory=list)
    prosecution: list[ProcessedEvidence] = field(default_factory=list)
    defense: list[ProcessedEvidence] = field(default_factory=list)
    moderator: list[ProcessedEvidence] = field(default_factory=list)

    _rag_index: Optional[object] = field(default=None, repr=False)
    _embedding_model: Optional[object] = field(default=None, repr=False)

    def _build_rag_index(self) -> None:
        """Build the RAG vector index from all chunks."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            print("Warning: chromadb not installed. RAG will not work.")
            return

        # Create in-memory collection
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        collection = client.create_collection(
            name="evidence",
            metadata={"hnsw:space": "cosine"}
        )

        # Add all chunks
        all_processed = self.shared + self.prosecution + self.defense + self.moderator

        documents = []
        metadatas = []
        ids = []

        for proc in all_processed:
            for i, chunk in enumerate(proc.chunks):
                doc_id = f"{proc.original.name}_{i}"
                documents.append(chunk)
                metadatas.append({
                    "source": proc.original.name,
                    "owner": proc.original.owner or "shared",
                    "chunk_index": i,
                })
                ids.append(doc_id)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        self._rag_index = collection

    def retrieve(
        self,
        query: str,
        role: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            role: Optional role to filter by visibility (prosecution, defense, moderator)
            top_k: Number of results to return

        Returns:
            List of {content, source, score} dicts
        """
        if not self._rag_index:
            return []

        k = top_k or self.config.top_k

        # Query the collection
        results = self._rag_index.query(
            query_texts=[query],
            n_results=k * 2,  # Get more, then filter
        )

        # Filter by visibility
        retrieved = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            owner = metadata.get("owner", "shared")

            # Check visibility
            if owner == "shared":
                visible = True
            elif role and owner == role:
                visible = True
            elif role == "moderator":
                # Moderator can see moderator docs
                visible = owner == "moderator"
            else:
                visible = False

            if visible:
                retrieved.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "score": 1 - distance,  # Convert distance to similarity
                    "owner": owner,
                })

            if len(retrieved) >= k:
                break

        return retrieved

    def get_context_for_role(
        self,
        role: str,
        query: Optional[str] = None,
    ) -> str:
        """
        Get processed evidence context for a role.

        For FULL/SUMMARIZE: Returns all visible processed content
        For RAG: Returns retrieved chunks based on query

        Args:
            role: The role (prosecution, defense, moderator)
            query: Optional query for RAG retrieval

        Returns:
            Formatted context string
        """
        from .agent import Role
        role_enum = Role(role) if isinstance(role, str) else role

        if self.config.strategy == ContextStrategy.RAG and query:
            return self._get_rag_context(role, query)

        # FULL or SUMMARIZE: return processed content
        visible = []
        visible.extend(self.shared)

        if role == "prosecution" or role_enum == Role.PROSECUTION:
            visible.extend(self.prosecution)
        elif role == "defense" or role_enum == Role.DEFENSE:
            visible.extend(self.defense)
        elif role == "moderator" or role_enum == Role.MODERATOR:
            visible.extend(self.moderator)

        if not visible:
            return ""

        sections = ["=== EVIDENCE ===\n"]
        for proc in visible:
            status = ""
            if proc.original.owner and proc.original.owner != "shared":
                status = " [PRIVATE]"
            if proc.summary:
                status += " [SUMMARIZED]"

            sections.append(f"--- {proc.original.name}{status} ---")
            sections.append(proc.processed_content)
            sections.append("")

        return "\n".join(sections)

    def _get_rag_context(self, role: str, query: str) -> str:
        """Get RAG-retrieved context for a query."""
        results = self.retrieve(query, role=role)

        if not results:
            return "No relevant evidence found for this query."

        sections = ["=== RETRIEVED EVIDENCE ===\n"]
        for r in results:
            sections.append(f"[From: {r['source']}] (relevance: {r['score']:.2f})")
            sections.append(r["content"])
            sections.append("")

        return "\n".join(sections)

    def get_shared_context(self) -> str:
        """Get context from shared evidence only (for opening statements)."""
        if not self.shared:
            return ""

        if self.config.strategy == ContextStrategy.RAG:
            # For RAG, just list available documents
            sections = ["=== SHARED EVIDENCE (available for retrieval) ===\n"]
            for proc in self.shared:
                sections.append(f"- {proc.original.name}")
            return "\n".join(sections)

        # FULL or SUMMARIZE: return processed shared content
        sections = ["=== SHARED EVIDENCE ===\n"]
        for proc in self.shared:
            status = ""
            if proc.summary:
                status = " [SUMMARIZED]"
            sections.append(f"--- {proc.original.name}{status} ---")
            sections.append(proc.processed_content)
            sections.append("")

        return "\n".join(sections)

    def total_tokens(self) -> int:
        """Estimate total tokens in processed evidence."""
        all_proc = self.shared + self.prosecution + self.defense + self.moderator
        return sum(p.token_estimate for p in all_proc)

    def summary(self) -> str:
        """Get a summary of processed evidence."""
        total_tokens = self.total_tokens()
        original_chars = self.original.total_chars()
        original_tokens = original_chars // 4

        lines = [
            f"Processed evidence ({self.config.strategy.value} strategy):",
            f"  Original: ~{original_tokens:,} tokens",
            f"  Processed: ~{total_tokens:,} tokens",
            f"  Reduction: {100 * (1 - total_tokens / max(original_tokens, 1)):.1f}%",
        ]

        if self.config.strategy == ContextStrategy.RAG:
            total_chunks = sum(
                len(p.chunks) for p in
                self.shared + self.prosecution + self.defense + self.moderator
            )
            lines.append(f"  RAG chunks: {total_chunks}")

        return "\n".join(lines)


def create_summarizer_from_provider(provider) -> Callable[[str], str]:
    """
    Create a summarizer function from an LLM provider.

    Args:
        provider: An LLMProvider instance

    Returns:
        A function that takes text and returns a summary
    """
    from ..providers import Message

    def summarize(text: str) -> str:
        messages = [Message(role="user", content=text)]
        response = provider.complete(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
        )
        return response.content

    return summarize
