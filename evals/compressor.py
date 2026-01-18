"""
Prompt Compressor Module

A multi-stage prompt compression system with LLMLingua-2 integration:
  Stage 0: Safe minification (whitespace, filler removal)
  Stage 1: Redundancy removal (exact + near-duplicate dedupe)
  Stage 2: ML compression via LLMLingua-2 (token-level learned compression)
  Stage 3: Relevance-based pruning (hybrid chunk selection) - fallback

Supports:
- ML-first compression: Pre-process -> LLMLingua-2 -> Post-process
- Hybrid compression (IR + embeddings + heuristics) as fallback
- Identity (no compression)

Public API:
    compress_text(prompt, importance_cutoff=..., mode="auto", **kwargs) -> (compressed, stats)
    compress_messages(messages, importance_cutoff=..., mode="auto", **kwargs) -> (compressed_messages, stats)
    identity_compressor(prompt, ...) -> (prompt, stats)
    count_tokens(text, encoding="o200k_base") -> int
"""

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set

# =============================================================================
# Optional Dependencies
# =============================================================================

TIKTOKEN_AVAILABLE = False
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    pass

LLMLINGUA_AVAILABLE = False
PromptCompressor = None
try:
    from llmlingua import PromptCompressor as _PromptCompressor
    PromptCompressor = _PromptCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    pass

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# =============================================================================
# Constants
# =============================================================================

SHORT_PROMPT_THRESHOLD = 350   # Tokens below which we use conservative compression
MEDIUM_PROMPT_THRESHOLD = 900  # Tokens below which we prefer paragraph-level
DEFAULT_CACHE_DIR = "runs/cache_compressed"

# LLMLingua-2 models (in order of preference)
LLMLINGUA2_MODELS = [
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
]
DEFAULT_ML_MODEL = LLMLINGUA2_MODELS[0]

# Tokens to always preserve in ML compression
FORCE_TOKENS_DEFAULT = [
    "\n", "?", ".", "!", ":", ";",
    "A.", "B.", "C.", "D.",
    "A)", "B)", "C)", "D)",
    "(A)", "(B)", "(C)", "(D)",
]

# Default min_reduction target
DEFAULT_MIN_REDUCTION = 0.05

# Near-duplicate similarity threshold
DEFAULT_NEAR_DUP_THRESHOLD = 0.97

# BERT max sequence length (LLMLingua-2 uses BERT-based models)
# Note: BERT uses WordPiece which can produce 2x+ more tokens than GPT's BPE for some text
# We use a very conservative limit to ensure chunks fit within BERT's 512 limit
BERT_MAX_TOKENS = 200  # Very conservative: 200 GPT tokens to stay well under BERT's 512


# =============================================================================
# Token Counting
# =============================================================================

class TokenCounter:
    """Token counter with tiktoken support and fallback heuristic."""

    def __init__(self, encoding_name: str = "o200k_base"):
        self.encoding_name = encoding_name
        self.encoder = None
        self.using_tiktoken = False

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
                self.using_tiktoken = True
            except Exception:
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                    self.using_tiktoken = True
                    self.encoding_name = "cl100k_base"
                except Exception:
                    pass

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self.using_tiktoken and self.encoder:
            return len(self.encoder.encode(text))
        return self._fallback_count(text)

    def _fallback_count(self, text: str) -> int:
        """Fallback token counter using heuristics."""
        if not text:
            return 0
        words = text.split()
        punct_count = len(re.findall(r"[.,!?;:'()\[\]{}]", text))
        return int((len(words) + punct_count) * 1.3)


_token_counters: Dict[str, TokenCounter] = {}


def get_token_counter(encoding: str = "o200k_base") -> TokenCounter:
    """Get or create token counter instance for encoding."""
    if encoding not in _token_counters:
        _token_counters[encoding] = TokenCounter(encoding)
    return _token_counters[encoding]


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Count tokens in text."""
    return get_token_counter(encoding).count(text)


# =============================================================================
# Compression Cache
# =============================================================================

class CompressionCache:
    """
    Cache for compressed outputs.

    Key: (prompt_hash, mode, importance_cutoff, target_tokens, target_reduction, model_name)
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Tuple[str, Dict]] = {}

    def _make_key(
        self,
        prompt_hash: str,
        mode: str,
        importance_cutoff: float,
        target_tokens: Optional[int],
        target_reduction: Optional[float],
        model_name: Optional[str],
    ) -> str:
        """Create cache key string."""
        parts = [
            prompt_hash,
            mode,
            f"cutoff={importance_cutoff:.4f}",
            f"target_tokens={target_tokens}" if target_tokens else "",
            f"target_reduction={target_reduction:.4f}" if target_reduction else "",
            f"model={model_name}" if model_name else "",
        ]
        key_str = "|".join(p for p in parts if p)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(
        self,
        prompt: str,
        mode: str,
        importance_cutoff: float,
        target_tokens: Optional[int] = None,
        target_reduction: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict]]:
        """Get cached compression result."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        key = self._make_key(prompt_hash, mode, importance_cutoff, target_tokens, target_reduction, model_name)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                result = (data["compressed"], data["stats"])
                self._memory_cache[key] = result
                return result
            except Exception:
                pass

        return None

    def set(
        self,
        prompt: str,
        mode: str,
        importance_cutoff: float,
        compressed: str,
        stats: Dict,
        target_tokens: Optional[int] = None,
        target_reduction: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Cache compression result."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        key = self._make_key(prompt_hash, mode, importance_cutoff, target_tokens, target_reduction, model_name)

        # Memory cache
        self._memory_cache[key] = (compressed, stats)

        # Disk cache
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump({"compressed": compressed, "stats": stats}, f)
        except Exception:
            pass


# Global cache instance
_compression_cache: Optional[CompressionCache] = None


def get_compression_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> CompressionCache:
    """Get or create compression cache."""
    global _compression_cache
    if _compression_cache is None:
        _compression_cache = CompressionCache(cache_dir)
    return _compression_cache


# =============================================================================
# Base Stats Dict
# =============================================================================

def make_stats(
    original_tokens: int,
    compressed_tokens: int,
    method: str,
    fallback_used: bool = False,
    error: str = "",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    rate: Optional[float] = None,
    target_tokens_used: Optional[int] = None,
    embeddings_used: bool = False,
    protected_spans_count: int = 0,
    chunks_total: int = 0,
    chunks_kept: int = 0,
    budget_tokens: int = 0,
    stage_used: Optional[List[str]] = None,
    min_reduction_target: float = DEFAULT_MIN_REDUCTION,
    dedupe_removed: int = 0,
    **extra,
) -> Dict[str, Any]:
    """
    Create standardized stats dictionary.

    Required fields:
        original_tokens, compressed_tokens, reduction_ratio, method, fallback_used, error
    Additional fields for eval compatibility:
        embeddings_used, protected_spans_count, chunks_total, chunks_kept,
        budget_tokens, stage_used, min_reduction_target, achieved_reduction
    """
    reduction = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0.0

    stats = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_ratio": reduction,
        "method": method,
        "fallback_used": fallback_used,
        "error": error,
        "embeddings_used": embeddings_used,
        "protected_spans_count": protected_spans_count,
        "chunks_total": chunks_total,
        "chunks_kept": chunks_kept,
        "budget_tokens": budget_tokens,
        "stage_used": stage_used or [],
        "min_reduction_target": min_reduction_target,
        "achieved_reduction": reduction,
        "dedupe_removed": dedupe_removed,
    }

    if model_name:
        stats["model_name"] = model_name
    if device:
        stats["device"] = device
    if rate is not None:
        stats["rate"] = rate
    if target_tokens_used is not None:
        stats["target_tokens"] = target_tokens_used

    stats.update(extra)
    return stats


# =============================================================================
# Stage 0: Minifier
# =============================================================================

class Minifier:
    """
    Stage 0: Safe minification.

    - Normalize whitespace (collapse multiple spaces/newlines)
    - Remove filler phrases
    - Trim trivial formatting noise
    """

    FILLER_PHRASES = [
        r"\bplease note that\b",
        r"\bit is important to note that\b",
        r"\bas mentioned earlier\b",
        r"\bin other words\b",
        r"\bthat being said\b",
        r"\bhaving said that\b",
        r"\bfor your information\b",
        r"\bbasically\b",
        r"\bessentially\b",
        r"\bactually\b",
        r"\bobviously\b",
        r"\bas you can see\b",
        r"\bas we know\b",
        r"\bit goes without saying\b",
        r"\bneedless to say\b",
        r"\bin fact\b",
        r"\bto be honest\b",
        r"\bfrankly speaking\b",
        r"\bas a matter of fact\b",
    ]

    def __init__(self):
        self.filler_regex = re.compile(
            "|".join(self.FILLER_PHRASES),
            re.IGNORECASE
        )

    def minify(self, text: str) -> str:
        """Apply safe minification to text."""
        if not text:
            return text

        # Remove filler phrases
        result = self.filler_regex.sub("", text)

        # Collapse multiple spaces to single space
        result = re.sub(r" +", " ", result)

        # Collapse 3+ newlines to 2
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Remove trailing spaces on lines
        result = re.sub(r" +\n", "\n", result)

        # Remove leading spaces on lines (but preserve code indentation by being conservative)
        # Only remove if the line starts with many spaces and isn't in a code block
        result = re.sub(r"^\s{4,}(?=[A-Za-z])", " ", result, flags=re.MULTILINE)

        return result.strip()


# =============================================================================
# Stage 1: Redundancy Removal
# =============================================================================

class RedundancyRemover:
    """
    Stage 1: Remove redundant content.

    - Exact duplicate removal at sentence/paragraph level
    - Near-duplicate removal via embeddings (if available)
    """

    def __init__(
        self,
        near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        use_embeddings: bool = True,
        max_units_for_embedding: int = 200,
    ):
        self.near_dup_threshold = near_dup_threshold
        self.use_embeddings = use_embeddings
        self.max_units_for_embedding = max_units_for_embedding
        self._embedding_model = None

    def _normalize_for_hash(self, text: str) -> str:
        """Normalize text for exact duplicate detection."""
        # Lowercase, strip, collapse whitespace
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with their positions."""
        sentences = []
        # Match sentences ending with .!? followed by space or end
        pattern = re.compile(r'[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$')
        for match in pattern.finditer(text):
            sent = match.group()
            if sent.strip():
                sentences.append((sent, match.start(), match.end()))
        return sentences

    def _split_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into paragraphs with their positions."""
        paragraphs = []
        # Split on double newlines
        parts = re.split(r'(\n\s*\n)', text)
        pos = 0
        for part in parts:
            if part.strip():
                paragraphs.append((part, pos, pos + len(part)))
            pos += len(part)
        return paragraphs

    def _get_embeddings(self, texts: List[str]) -> Optional[Any]:
        """Compute embeddings for texts if available."""
        if not self.use_embeddings or SentenceTransformer is None or np is None:
            return None

        try:
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embedding_model.encode(texts, normalize_embeddings=True)
        except Exception:
            return None

    def _cosine_similarity(self, emb1: Any, emb2: Any) -> float:
        """Compute cosine similarity between two embeddings."""
        if np is None:
            return 0.0
        # Already normalized, so dot product = cosine similarity
        return float(np.dot(emb1, emb2))

    def _dedupe_exact(self, units: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """Remove exact duplicates, keeping first occurrence."""
        seen_hashes: Set[str] = set()
        result = []
        for text, start, end in units:
            h = self._normalize_for_hash(text)
            if h and h not in seen_hashes:
                seen_hashes.add(h)
                result.append((text, start, end))
        return result

    def _dedupe_near(self, units: List[Tuple[str, int, int]]) -> Tuple[List[Tuple[str, int, int]], bool]:
        """
        Remove near-duplicates using embeddings.
        Returns (deduplicated_units, embeddings_used).
        """
        if len(units) <= 1:
            return units, False

        # Skip if too many units
        if len(units) > self.max_units_for_embedding:
            return units, False

        texts = [u[0] for u in units]
        embeddings = self._get_embeddings(texts)

        if embeddings is None:
            return units, False

        # Mark indices to keep (keep first, drop later near-duplicates)
        keep_mask = [True] * len(units)

        for i in range(len(units)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(units)):
                if not keep_mask[j]:
                    continue
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self.near_dup_threshold:
                    keep_mask[j] = False

        result = [units[i] for i in range(len(units)) if keep_mask[i]]
        return result, True

    def dedupe(
        self,
        text: str,
        level: str = "auto",
        token_counter: Optional[TokenCounter] = None,
    ) -> Tuple[str, int, bool]:
        """
        Remove redundant content from text.

        Args:
            text: Input text
            level: "sentence", "paragraph", or "auto" (chooses based on text length)
            token_counter: Token counter for length estimation

        Returns:
            (deduplicated_text, units_removed, embeddings_used)
        """
        if not text or not text.strip():
            return text, 0, False

        # Determine level automatically
        if level == "auto":
            if token_counter:
                tokens = token_counter.count(text)
            else:
                tokens = len(text.split()) * 1.3

            if tokens <= SHORT_PROMPT_THRESHOLD:
                level = "sentence"
            elif tokens <= MEDIUM_PROMPT_THRESHOLD:
                level = "paragraph"
            else:
                level = "sentence"  # For long texts, sentence-level gives more granular dedupe

        # Split into units
        if level == "paragraph":
            units = self._split_paragraphs(text)
        else:
            units = self._split_sentences(text)

        if not units:
            return text, 0, False

        original_count = len(units)

        # Stage 1a: Exact dedupe
        units = self._dedupe_exact(units)

        # Stage 1b: Near-duplicate dedupe
        units, embeddings_used = self._dedupe_near(units)

        removed_count = original_count - len(units)

        # Reconstruct text preserving order
        if removed_count == 0:
            return text, 0, embeddings_used

        # Sort by original position
        units.sort(key=lambda x: x[1])

        # Join with appropriate separator
        if level == "paragraph":
            result = "\n\n".join(u[0].strip() for u in units)
        else:
            result = " ".join(u[0].strip() for u in units)

        return result, removed_count, embeddings_used


# =============================================================================
# Protected Span Extraction
# =============================================================================

class ProtectedSpanExtractor:
    """Extract and restore protected spans using placeholders."""

    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    URL_RE = re.compile(r"https?://[^\s]+")
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[\w\-._~/:\\]+")
    DATE_RE = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b")
    LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
    HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b")
    UUID_RE = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
    )
    DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
    QUOTED_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
    ANSWER_WITH_RE = re.compile(
        r"answer\s+with\s+(?:[A-D](?:\s*/\s*[A-D]){1,3}|[A-D](?:\s*,\s*[A-D]){1,3})",
        re.IGNORECASE,
    )
    # Also protect "Answer with A/B/C/D" style instructions at end
    ANSWER_INSTRUCTION_RE = re.compile(
        r"(?:Answer|Select|Choose)\s+(?:with\s+)?(?:option\s+)?[A-D](?:[,/\s]+(?:or\s+)?[A-D])*\.?",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> Tuple[str, Dict[str, str], int]:
        """Extract protected spans and replace with placeholders."""
        spans = []
        spans.extend(self._find_code_blocks(text))
        spans.extend(self._find_json_blocks(text))
        spans.extend(self._find_by_regex(self.URL_RE, text))
        spans.extend(self._find_by_regex(self.EMAIL_RE, text))
        spans.extend(self._find_by_regex(self.PATH_RE, text))
        spans.extend(self._find_by_regex(self.DATE_RE, text))
        spans.extend(self._find_by_regex(self.LONG_NUMBER_RE, text))
        spans.extend(self._find_by_regex(self.HEX_RE, text))
        spans.extend(self._find_by_regex(self.UUID_RE, text))
        spans.extend(self._find_by_regex(self.DECIMAL_RE, text))
        spans.extend(self._find_by_regex(self.QUOTED_RE, text))
        spans.extend(self._find_by_regex(self.ANSWER_WITH_RE, text))
        spans.extend(self._find_by_regex(self.ANSWER_INSTRUCTION_RE, text))

        if not spans:
            return text, {}, 0

        merged_spans = self._merge_spans(spans)
        placeholders = {}
        output = []
        last = 0
        for i, (start, end) in enumerate(merged_spans):
            output.append(text[last:start])
            key = f"__PROTECTED_{i}__"
            placeholders[key] = text[start:end]
            output.append(key)
            last = end
        output.append(text[last:])

        return "".join(output), placeholders, len(merged_spans)

    def restore(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore protected spans from placeholders."""
        restored = text
        for key, value in placeholders.items():
            restored = restored.replace(key, value)
        return restored

    def _find_by_regex(self, pattern: re.Pattern, text: str) -> List[Tuple[int, int]]:
        return [(m.start(), m.end()) for m in pattern.finditer(text)]

    def _find_code_blocks(self, text: str) -> List[Tuple[int, int]]:
        return [(m.start(), m.end()) for m in self.CODE_BLOCK_RE.finditer(text)]

    def _find_json_blocks(self, text: str) -> List[Tuple[int, int]]:
        """Find JSON/YAML-like blocks."""
        spans = []
        for brace_pair in [('{', '}'), ('[', ']')]:
            depth = 0
            start_idx = None
            for i, ch in enumerate(text):
                if ch == brace_pair[0]:
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == brace_pair[1]:
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_idx is not None:
                            # Only add if it looks like JSON (has : or ,)
                            block = text[start_idx:i+1]
                            if ':' in block or ',' in block:
                                spans.append((start_idx, i + 1))
                            start_idx = None
        return spans

    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping spans."""
        unique = sorted(set(spans), key=lambda s: (s[0], s[1]))
        if not unique:
            return []
        merged = [list(unique[0])]
        for start, end in unique[1:]:
            last = merged[-1]
            if start <= last[1]:
                if end > last[1]:
                    last[1] = end
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]


# =============================================================================
# Keyword Extraction
# =============================================================================

class KeywordExtractor:
    """Extract important keywords/entities from query text."""

    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'although', 'though', 'after', 'before',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
        'it', 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
        'theirs', 'themselves', 'answer', 'question', 'following', 'please',
    }

    def extract(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract top keywords from text."""
        if not text:
            return []

        # Tokenize
        tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', text.lower())

        # Filter stopwords and short tokens
        tokens = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]

        # Count frequencies
        freq = Counter(tokens)

        # Return top keywords
        return [word for word, _ in freq.most_common(max_keywords)]


# =============================================================================
# Chunking
# =============================================================================

@dataclass
class TextChunk:
    text: str
    start_idx: int
    end_idx: int
    token_count: int
    score: float = 0.0
    has_protected: bool = False
    has_instruction: bool = False


class Chunker:
    """Split text into chunks for relevance scoring."""

    def __init__(self, token_counter: TokenCounter, max_paragraph_tokens: int = 120):
        self.token_counter = token_counter
        self.max_paragraph_tokens = max_paragraph_tokens

    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into chunks."""
        chunks: List[TextChunk] = []
        for start, end in self._paragraph_spans(text):
            para_text = text[start:end].strip()
            if not para_text:
                continue
            tokens = self.token_counter.count(para_text)
            if tokens > self.max_paragraph_tokens:
                chunks.extend(self._sentence_chunks(para_text, start))
            else:
                chunks.append(TextChunk(
                    text=para_text, start_idx=start, end_idx=end, token_count=tokens
                ))
        return chunks

    def _paragraph_spans(self, text: str) -> List[Tuple[int, int]]:
        spans = []
        start = 0
        for match in re.finditer(r"\n\s*\n", text):
            end = match.start()
            if text[start:end].strip():
                spans.append((start, end))
            start = match.end()
        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))
        return spans

    def _sentence_chunks(self, paragraph: str, offset: int) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        sentence_re = re.compile(r"[^.!?]+(?:[.!?]+|$)\s*", re.MULTILINE)
        for match in sentence_re.finditer(paragraph):
            sent = match.group().strip()
            if not sent:
                continue
            start_idx = offset + match.start()
            end_idx = offset + match.end()
            chunks.append(TextChunk(
                text=sent, start_idx=start_idx, end_idx=end_idx,
                token_count=self.token_counter.count(sent)
            ))
        return chunks


class SemanticChunker:
    """
    Smart chunker for LLMLingua-2 that respects BERT token limits.

    Splits at semantic boundaries (paragraphs, sentences) while keeping
    chunks under BERT's max sequence length.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = BERT_MAX_TOKENS,
    ):
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks that fit within BERT's context window."""
        total_tokens = self.token_counter.count(text)

        if total_tokens <= self.max_tokens:
            return [text]

        chunks = []

        # Try paragraph-level first
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count(para)

            # If single paragraph exceeds limit, split by sentences
            if para_tokens > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph by sentences
                sentence_chunks = self._split_by_sentences(para)
                chunks.extend(sentence_chunks)

            elif current_tokens + para_tokens + 2 > self.max_tokens:  # +2 for newlines
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

            else:
                current_chunk.append(para)
                current_tokens += para_tokens + 2

        # Flush remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences, respecting token limits."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_tokens = self.token_counter.count(sent)

            # If single sentence exceeds limit, truncate it
            if sent_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                # Truncate long sentence by words
                truncated = self._truncate_to_limit(sent)
                chunks.append(truncated)

            elif current_tokens + sent_tokens + 1 > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens

            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _truncate_to_limit(self, text: str) -> str:
        """Truncate text to fit within max_tokens limit."""
        words = text.split()
        if not words:
            return text

        # Binary search for the right length
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = " ".join(words[:mid])
            if self.token_counter.count(candidate) <= self.max_tokens:
                lo = mid
            else:
                hi = mid - 1

        return " ".join(words[:lo])


# =============================================================================
# Relevance Scoring
# =============================================================================

class RelevanceScorer:
    """Score chunks by relevance using embeddings, BM25, or lexical overlap."""

    def __init__(self, use_embeddings: bool = True, cache_dir: str = ".cache/compressor"):
        self.use_embeddings = use_embeddings
        self.cache_dir = cache_dir
        self._embedding_model = None

    def score(self, query: str, chunks: List[TextChunk]) -> Tuple[List[float], str, bool]:
        """Score chunks and return (scores, method, embeddings_used)."""
        if not chunks:
            return [], "lexical", False

        if self.use_embeddings and SentenceTransformer is not None and np is not None:
            try:
                return self._embedding_score(query, chunks), "hybrid_embeddings", True
            except Exception:
                pass

        if BM25Okapi is not None:
            try:
                return self._bm25_score(query, chunks), "bm25", False
            except Exception:
                pass

        return self._lexical_score(query, chunks), "lexical", False

    def _embedding_score(self, query: str, chunks: List[TextChunk]) -> List[float]:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [query] + [c.text for c in chunks]
        embeddings = self._embedding_model.encode(texts, normalize_embeddings=True)
        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]
        scores = chunk_embs @ query_emb
        return [float(s) for s in scores]

    def _bm25_score(self, query: str, chunks: List[TextChunk]) -> List[float]:
        tokenized = [self._tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self._tokenize(query))
        return [float(s) for s in scores]

    def _lexical_score(self, query: str, chunks: List[TextChunk]) -> List[float]:
        query_tokens = set(self._tokenize(query))
        scores = []
        for chunk in chunks:
            chunk_tokens = set(self._tokenize(chunk.text))
            overlap = len(chunk_tokens & query_tokens)
            scores.append(overlap / ((chunk.token_count + 10) ** 0.5))
        return scores

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_\-/]+", text.lower())


# =============================================================================
# Budget Selection (Stage 3 - Fallback)
# =============================================================================

class BudgetSelector:
    """Select chunks within token budget based on relevance scores."""

    # Instruction/question markers
    INSTRUCTION_MARKERS = (
        "question:", "q:", "instruction:", "task:", "query:",
        "answer with", "select", "choose", "what is", "what was",
        "how did", "why did", "which", "who", "where", "when",
    )

    def __init__(self, keyword_extractor: Optional[KeywordExtractor] = None):
        self.keyword_extractor = keyword_extractor or KeywordExtractor()

    def select(
        self,
        chunks: List[TextChunk],
        budget_tokens: int,
        query_text: str,
        require_keywords: bool = True,
    ) -> List[TextChunk]:
        """
        Select best chunks within budget.

        Key change: Protected chunks are NOT automatically mandatory.
        We do ensure instruction-containing chunks are kept.
        """
        if not chunks:
            return []

        # Mark instruction chunks
        self._mark_instruction_chunks(chunks)

        # Get required keywords from query
        required_keywords = []
        if require_keywords and query_text:
            required_keywords = self.keyword_extractor.extract(query_text, max_keywords=5)

        # Find chunks that contain required keywords
        keyword_chunks: Set[int] = set()
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.text.lower()
            for kw in required_keywords:
                if kw in chunk_lower:
                    keyword_chunks.add(i)
                    break

        selected_idx: List[int] = []
        used_tokens = 0

        # First pass: include instruction chunks (they contain the question)
        for i, chunk in enumerate(chunks):
            if chunk.has_instruction:
                if used_tokens + chunk.token_count <= budget_tokens:
                    selected_idx.append(i)
                    used_tokens += chunk.token_count

        # Second pass: include keyword-covering chunks
        for i in keyword_chunks:
            if i in selected_idx:
                continue
            chunk = chunks[i]
            if used_tokens + chunk.token_count <= budget_tokens:
                selected_idx.append(i)
                used_tokens += chunk.token_count

        # Third pass: add by score until budget filled
        remaining = [
            (i, chunks[i]) for i in range(len(chunks))
            if i not in set(selected_idx)
        ]
        # Sort by score descending, then by position for stability
        remaining.sort(key=lambda x: (-x[1].score, x[0]))

        for i, chunk in remaining:
            if used_tokens + chunk.token_count <= budget_tokens:
                selected_idx.append(i)
                used_tokens += chunk.token_count

        # Return in original order
        selected_idx = sorted(set(selected_idx))

        # Guarantee non-empty: include at least highest-scoring or last chunk
        if not selected_idx:
            # Find highest scoring chunk that fits
            best_idx = max(range(len(chunks)), key=lambda i: (chunks[i].score, -i))
            selected_idx = [best_idx]

        return [chunks[i] for i in selected_idx]

    def _mark_instruction_chunks(self, chunks: List[TextChunk]) -> None:
        """Mark chunks that contain instruction/question content."""
        for chunk in chunks:
            lower = chunk.text.lower()
            if any(marker in lower for marker in self.INSTRUCTION_MARKERS):
                chunk.has_instruction = True
            elif "?" in chunk.text:
                chunk.has_instruction = True
            # Also check for A/B/C/D answer options
            elif re.search(r'\b[A-D]\.\s', chunk.text):
                chunk.has_instruction = True


# =============================================================================
# Instruction/Context Segmentation
# =============================================================================

class TextSegmenter:
    """Segment text into instruction vs context regions."""

    CONTEXT_MARKERS = [
        "context:", "background:", "information:", "details:",
        "the following", "given:", "provided:", "based on",
        "read the", "passage:", "document:", "text:",
    ]

    INSTRUCTION_MARKERS = [
        "question:", "q:", "instruction:", "task:", "answer",
        "select", "choose", "what", "which", "who", "where",
        "when", "how", "why", "based on the above",
    ]

    def segment(self, text: str) -> Tuple[str, str, str]:
        """
        Segment text into (preamble, context, instruction).

        Returns empty strings for parts that couldn't be identified.
        """
        lines = text.split('\n')

        # Find where context starts and ends
        context_start = -1
        instruction_start = len(lines)

        for i, line in enumerate(lines):
            lower = line.lower().strip()

            # Check for context markers
            if context_start == -1:
                for marker in self.CONTEXT_MARKERS:
                    if lower.startswith(marker) or marker in lower[:50]:
                        context_start = i
                        break

            # Check for instruction markers (usually near the end)
            for marker in self.INSTRUCTION_MARKERS:
                if marker in lower:
                    # Only update if we're in the latter half of text
                    if i > len(lines) * 0.5:
                        instruction_start = min(instruction_start, i)
                        break

        # If no clear markers, use heuristic: last paragraph is instruction
        if instruction_start == len(lines):
            # Find last paragraph break
            for i in range(len(lines) - 1, -1, -1):
                if not lines[i].strip():
                    instruction_start = i + 1
                    break

        # Build segments
        if context_start == -1:
            context_start = 0

        preamble = '\n'.join(lines[:context_start]) if context_start > 0 else ""
        context = '\n'.join(lines[context_start:instruction_start])
        instruction = '\n'.join(lines[instruction_start:])

        return preamble.strip(), context.strip(), instruction.strip()


# =============================================================================
# LLMLingua-2 Compressor (ML Stage)
# =============================================================================

class LLMLingua2Compressor:
    """
    LLMLingua-2 based compressor with smart pre/post-processing.

    Pipeline:
    1. Extract protected spans
    2. Pre-process (minify, dedupe)
    3. Smart chunking for BERT limits
    4. LLMLingua-2 compression per chunk
    5. Restore protected spans
    6. Post-process whitespace
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ML_MODEL,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._compressor: Optional[Any] = None
        self._init_error: Optional[str] = None
        self._protector = ProtectedSpanExtractor()
        self._minifier = Minifier()

    def _load_compressor(self) -> bool:
        """Lazy load the LLMLingua-2 compressor."""
        if self._compressor is not None:
            return True
        if self._init_error is not None:
            return False

        if not LLMLINGUA_AVAILABLE:
            self._init_error = "llmlingua not installed. Install with: pip install llmlingua"
            return False

        try:
            # Suppress transformer warnings about sequence length
            import warnings
            import logging as _logging
            _logging.getLogger("transformers").setLevel(_logging.ERROR)
            warnings.filterwarnings("ignore", message=".*sequence length.*")
            warnings.filterwarnings("ignore", message=".*torch_dtype.*")

            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
                device_map=self.device,
            )
            return True
        except Exception as e:
            self._init_error = f"Failed to load LLMLingua-2 model: {e}"
            return False

    def is_available(self) -> bool:
        """Check if LLMLingua-2 is available."""
        return self._load_compressor()

    def compress(
        self,
        text: str,
        rate: float = 0.5,
        force_tokens: Optional[List[str]] = None,
        token_counter: Optional[TokenCounter] = None,
        use_preprocess: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress text using LLMLingua-2.

        Args:
            text: Input text
            rate: Keep rate (0-1), e.g., 0.5 keeps ~50% of tokens
            force_tokens: Tokens to always preserve
            token_counter: Token counter for stats
            use_preprocess: Whether to apply minify/dedupe first

        Returns:
            (compressed_text, stats_dict)
        """
        if token_counter is None:
            token_counter = get_token_counter()

        original_tokens = token_counter.count(text)

        if not text or original_tokens == 0:
            return text, make_stats(0, 0, "llmlingua2", error="empty input")

        if not self._load_compressor():
            return text, make_stats(
                original_tokens, original_tokens, "llmlingua2",
                fallback_used=True, error=self._init_error or "Unknown error",
                model_name=self.model_name, device=self.device
            )

        stages_used = []
        protected_count = 0
        dedupe_removed = 0

        try:
            # Step 1: Extract protected spans
            working_text, placeholders, protected_count = self._protector.extract(text)

            # Step 2: Pre-process (optional but recommended)
            if use_preprocess:
                # Minify
                stages_used.append("minify")
                working_text = self._minifier.minify(working_text)

                # Dedupe
                stages_used.append("dedupe")
                redundancy_remover = RedundancyRemover(use_embeddings=False)  # Fast exact dedupe only
                working_text, dedupe_removed, _ = redundancy_remover.dedupe(
                    working_text, level="sentence", token_counter=token_counter
                )

            # Step 3: Smart chunking
            stages_used.append("ml_compress")
            chunker = SemanticChunker(token_counter)
            chunks = chunker.chunk(working_text)

            # Step 4: Compress each chunk with LLMLingua-2
            if force_tokens is None:
                force_tokens = FORCE_TOKENS_DEFAULT

            compressed_chunks = []
            for chunk in chunks:
                try:
                    # LLMLingua-2 compress_prompt with parameters to handle chunking
                    result = self._compressor.compress_prompt(
                        chunk,
                        rate=rate,
                        force_tokens=force_tokens,
                        drop_consecutive=True,
                        chunk_end_tokens=["."],  # Split on sentence boundaries
                        use_context_level_filter=False,  # Disable context-level to avoid memory issues
                    )
                    compressed_chunks.append(result.get("compressed_prompt", chunk))
                except Exception:
                    # If chunk fails (e.g., sequence too long), keep original chunk
                    compressed_chunks.append(chunk)

            # Step 5: Reassemble
            compressed_text = "\n\n".join(compressed_chunks)

            # Step 6: Restore protected spans
            compressed_text = self._protector.restore(compressed_text, placeholders)

            # Step 7: Normalize whitespace
            compressed_text = self._normalize_whitespace(compressed_text)

            compressed_tokens = token_counter.count(compressed_text)

            # Never expand text
            if compressed_tokens >= original_tokens:
                return text, make_stats(
                    original_tokens, original_tokens, "llmlingua2",
                    fallback_used=True,
                    error="Compression would expand text",
                    model_name=self.model_name, device=self.device,
                    rate=rate, stage_used=stages_used,
                    protected_spans_count=protected_count,
                )

            return compressed_text, make_stats(
                original_tokens, compressed_tokens, "llmlingua2",
                model_name=self.model_name, device=self.device,
                rate=rate, stage_used=stages_used,
                protected_spans_count=protected_count,
                dedupe_removed=dedupe_removed,
                chunks_total=len(chunks),
            )

        except Exception as e:
            return text, make_stats(
                original_tokens, original_tokens, "llmlingua2",
                fallback_used=True, error=str(e),
                model_name=self.model_name, device=self.device,
                stage_used=stages_used,
            )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in output."""
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# =============================================================================
# Hybrid Compressor (Fallback)
# =============================================================================

class HybridCompressor:
    """
    Multi-stage hybrid compressor (fallback when LLMLingua-2 unavailable).

    Stage 0: Safe minification
    Stage 1: Redundancy removal (exact + near-duplicate)
    Stage 2: Relevance-based chunk selection
    """

    def __init__(self, encoding: str = "o200k_base"):
        self.encoding = encoding
        self.protector = ProtectedSpanExtractor()
        self.minifier = Minifier()
        self.segmenter = TextSegmenter()
        self.keyword_extractor = KeywordExtractor()

    def compress(
        self,
        text: str,
        importance_cutoff: float = 0.5,
        target_tokens: Optional[int] = None,
        target_reduction: Optional[float] = None,
        use_embeddings: bool = True,
        cache_dir: str = ".cache/compressor",
        query_text: Optional[str] = None,
        encoding: Optional[str] = None,
        min_reduction: float = DEFAULT_MIN_REDUCTION,
        near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        dedupe_level: str = "auto",
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress using multi-stage pipeline.

        Stage 0: Minification (always runs)
        Stage 1: Dedupe (always runs)
        Stage 2: Chunk selection (runs if needed to hit min_reduction)
        """
        encoding = encoding or self.encoding
        token_counter = get_token_counter(encoding)
        original_tokens = token_counter.count(text)

        if not text or original_tokens == 0:
            return text, make_stats(0, 0, "hybrid")

        stages_used: List[str] = []
        embeddings_used = False
        protected_count = 0
        chunks_total = 0
        chunks_kept = 0
        dedupe_removed = 0

        try:
            # Extract protected spans FIRST
            working_text, placeholders, protected_count = self.protector.extract(text)

            # ===== Stage 0: Minification =====
            stages_used.append("minify")
            working_text = self.minifier.minify(working_text)

            # ===== Stage 1: Redundancy Removal =====
            stages_used.append("dedupe")
            redundancy_remover = RedundancyRemover(
                near_dup_threshold=near_dup_threshold,
                use_embeddings=use_embeddings,
            )

            # Determine dedupe level based on prompt length
            if dedupe_level == "auto":
                if original_tokens <= SHORT_PROMPT_THRESHOLD:
                    dedupe_level = "sentence"
                elif original_tokens <= MEDIUM_PROMPT_THRESHOLD:
                    dedupe_level = "paragraph"
                else:
                    dedupe_level = "sentence"

            working_text, dedupe_removed, dedupe_emb_used = redundancy_remover.dedupe(
                working_text,
                level=dedupe_level,
                token_counter=token_counter,
            )
            embeddings_used = dedupe_emb_used

            # Check reduction after Stage 1
            deduped_with_protected = self.protector.restore(working_text, placeholders)
            tokens_after_dedupe = token_counter.count(deduped_with_protected)
            reduction_after_dedupe = (original_tokens - tokens_after_dedupe) / original_tokens if original_tokens > 0 else 0

            # ===== Stage 2: Relevance Selection (if needed) =====
            # Only apply if we haven't hit min_reduction AND text is long enough
            need_stage2 = (
                reduction_after_dedupe < min_reduction and
                original_tokens > SHORT_PROMPT_THRESHOLD
            )

            # Also apply stage 2 if explicitly requested via high cutoff or target
            if target_tokens is not None or target_reduction is not None or importance_cutoff >= 0.5:
                need_stage2 = True

            if need_stage2:
                stages_used.append("select")

                # Compute budget
                budget_tokens = self._compute_budget(
                    original_tokens, importance_cutoff, target_tokens, target_reduction
                )

                # Extract query if not provided
                if query_text is None:
                    query_text = self._extract_query(text)

                # Try to segment and only compress context
                preamble, context, instruction = self.segmenter.segment(working_text)

                if context and len(context) > len(instruction):
                    # Compress only the context portion
                    chunker = Chunker(token_counter)
                    chunks = chunker.chunk(context)

                    if chunks:
                        chunks_total = len(chunks)

                        # Mark protected chunks
                        placeholder_re = re.compile(r"__PROTECTED_\d+__")
                        for chunk in chunks:
                            chunk.has_protected = bool(placeholder_re.search(chunk.text))

                        # Score chunks
                        scorer = RelevanceScorer(use_embeddings=use_embeddings, cache_dir=cache_dir)
                        scores, method, score_emb_used = scorer.score(query_text, chunks)
                        embeddings_used = embeddings_used or score_emb_used

                        for chunk, score in zip(chunks, scores):
                            chunk.score = score

                        # Calculate context budget
                        preamble_tokens = token_counter.count(preamble) if preamble else 0
                        instruction_tokens = token_counter.count(instruction) if instruction else 0
                        context_budget = max(1, budget_tokens - preamble_tokens - instruction_tokens)

                        # Select chunks
                        selector = BudgetSelector(self.keyword_extractor)
                        selected = selector.select(chunks, context_budget, query_text)
                        chunks_kept = len(selected)

                        # Reassemble
                        compressed_context = "\n\n".join(
                            chunk.text.strip() for chunk in selected if chunk.text.strip()
                        )

                        # Rebuild full text
                        parts = []
                        if preamble:
                            parts.append(preamble)
                        if compressed_context:
                            parts.append(compressed_context)
                        if instruction:
                            parts.append(instruction)

                        working_text = "\n\n".join(parts)
                else:
                    # Fall back to chunking whole text
                    chunker = Chunker(token_counter)
                    chunks = chunker.chunk(working_text)

                    if chunks:
                        chunks_total = len(chunks)

                        # Mark protected chunks
                        placeholder_re = re.compile(r"__PROTECTED_\d+__")
                        for chunk in chunks:
                            chunk.has_protected = bool(placeholder_re.search(chunk.text))

                        # Score chunks
                        scorer = RelevanceScorer(use_embeddings=use_embeddings, cache_dir=cache_dir)
                        scores, method, score_emb_used = scorer.score(query_text, chunks)
                        embeddings_used = embeddings_used or score_emb_used

                        for chunk, score in zip(chunks, scores):
                            chunk.score = score

                        # Select chunks
                        selector = BudgetSelector(self.keyword_extractor)
                        selected = selector.select(chunks, budget_tokens, query_text)
                        chunks_kept = len(selected)

                        # Reassemble
                        working_text = "\n\n".join(
                            chunk.text.strip() for chunk in selected if chunk.text.strip()
                        )

            # Restore protected spans
            output_text = self.protector.restore(working_text, placeholders)
            output_text = self._normalize_whitespace(output_text)

            if not output_text.strip():
                raise RuntimeError("Empty output after compression")

            compressed_tokens = token_counter.count(output_text)

            # Never expand
            if compressed_tokens >= original_tokens:
                return text, make_stats(
                    original_tokens, original_tokens, "hybrid",
                    fallback_used=True, error="Compression would expand",
                    embeddings_used=embeddings_used,
                    protected_spans_count=protected_count,
                    stage_used=stages_used,
                    min_reduction_target=min_reduction,
                )

            # Determine method based on stages
            if "select" in stages_used:
                method = "hybrid_embeddings" if embeddings_used else "bm25" if BM25Okapi else "lexical"
            elif dedupe_removed > 0:
                method = "dedupe"
            else:
                method = "minify"

            return output_text, make_stats(
                original_tokens, compressed_tokens, method,
                embeddings_used=embeddings_used,
                protected_spans_count=protected_count,
                chunks_total=chunks_total,
                chunks_kept=chunks_kept,
                budget_tokens=self._compute_budget(original_tokens, importance_cutoff, target_tokens, target_reduction) if need_stage2 else 0,
                stage_used=stages_used,
                min_reduction_target=min_reduction,
                dedupe_removed=dedupe_removed,
            )

        except Exception as e:
            # Fail-open: return original or minified text
            try:
                safe_text = self.minifier.minify(text)
                safe_tokens = token_counter.count(safe_text)
                if safe_tokens < original_tokens:
                    return safe_text, make_stats(
                        original_tokens, safe_tokens, "minify",
                        fallback_used=True, error=str(e),
                        stage_used=["minify"],
                        min_reduction_target=min_reduction,
                    )
            except Exception:
                pass

            return text, make_stats(
                original_tokens, original_tokens, "hybrid",
                fallback_used=True, error=str(e),
                stage_used=stages_used,
                min_reduction_target=min_reduction,
            )

    def _compute_budget(
        self,
        original_tokens: int,
        importance_cutoff: float,
        target_tokens: Optional[int],
        target_reduction: Optional[float],
    ) -> int:
        """Compute token budget for selection."""
        if target_tokens is not None:
            return max(1, min(original_tokens, int(target_tokens)))
        if target_reduction is not None:
            return max(1, int(original_tokens * (1 - target_reduction)))
        # Use importance_cutoff: budget = original * (1 - cutoff * 0.6)
        cutoff = max(0.0, min(1.0, float(importance_cutoff)))
        budget = int(original_tokens * (1 - cutoff * 0.6))
        return max(1, min(original_tokens, budget))

    def _extract_query(self, text: str) -> str:
        """Extract query/question from text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        markers = ("question:", "q:", "instruction:", "task:", "query:")
        for idx in range(len(lines) - 1, -1, -1):
            lower = lines[idx].lower()
            if lower.startswith(markers) or "?" in lines[idx]:
                return lines[idx]
        return text.strip()[-1000:]

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in output."""
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# =============================================================================
# Unified Compressor (ML-first with Hybrid fallback)
# =============================================================================

class UnifiedCompressor:
    """
    Unified compressor that uses LLMLingua-2 when available, falls back to hybrid.

    Pipeline:
    1. Try LLMLingua-2 (ML compression)
    2. If unavailable/fails, use hybrid compression
    3. Report which method was used in stats
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ML_MODEL,
        device: str = "cpu",
        encoding: str = "o200k_base",
    ):
        self.model_name = model_name
        self.device = device
        self.encoding = encoding
        self._ml_compressor: Optional[LLMLingua2Compressor] = None
        self._hybrid_compressor: Optional[HybridCompressor] = None

    def _get_ml_compressor(self) -> LLMLingua2Compressor:
        if self._ml_compressor is None:
            self._ml_compressor = LLMLingua2Compressor(
                model_name=self.model_name,
                device=self.device,
            )
        return self._ml_compressor

    def _get_hybrid_compressor(self) -> HybridCompressor:
        if self._hybrid_compressor is None:
            self._hybrid_compressor = HybridCompressor(encoding=self.encoding)
        return self._hybrid_compressor

    def compress(
        self,
        text: str,
        importance_cutoff: float = 0.5,
        target_tokens: Optional[int] = None,
        target_reduction: Optional[float] = None,
        use_embeddings: bool = True,
        query_text: Optional[str] = None,
        min_reduction: float = DEFAULT_MIN_REDUCTION,
        force_tokens: Optional[List[str]] = None,
        prefer_ml: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress text using best available method.

        Args:
            text: Input text to compress
            importance_cutoff: 0-1, higher = more aggressive
            target_tokens: Target token count (optional)
            target_reduction: Target reduction ratio 0-1 (optional)
            use_embeddings: Use embeddings for hybrid scoring
            query_text: Query for relevance scoring
            min_reduction: Minimum reduction target
            force_tokens: Tokens to preserve in ML mode
            prefer_ml: Prefer ML compression when available

        Returns:
            (compressed_text, stats_dict)
        """
        token_counter = get_token_counter(self.encoding)
        original_tokens = token_counter.count(text)

        if not text or original_tokens == 0:
            return text, make_stats(0, 0, "identity")

        # Compute rate for ML compression
        rate = self._compute_rate(
            original_tokens, importance_cutoff, target_tokens, target_reduction
        )

        # Try ML compression first
        if prefer_ml:
            ml_compressor = self._get_ml_compressor()
            if ml_compressor.is_available():
                compressed, stats = ml_compressor.compress(
                    text,
                    rate=rate,
                    force_tokens=force_tokens,
                    token_counter=token_counter,
                    use_preprocess=True,
                )

                # Check if ML compression succeeded
                if not stats.get("fallback_used"):
                    stats["compression_method"] = "llmlingua2"
                    return compressed, stats

        # Fall back to hybrid compression
        hybrid_compressor = self._get_hybrid_compressor()
        compressed, stats = hybrid_compressor.compress(
            text,
            importance_cutoff=importance_cutoff,
            target_tokens=target_tokens,
            target_reduction=target_reduction,
            use_embeddings=use_embeddings,
            query_text=query_text,
            min_reduction=min_reduction,
            encoding=self.encoding,
            **kwargs,
        )

        stats["compression_method"] = "hybrid"
        if prefer_ml and LLMLINGUA_AVAILABLE:
            stats["ml_fallback_reason"] = "LLMLingua-2 compression failed or unavailable"

        return compressed, stats

    def _compute_rate(
        self,
        original_tokens: int,
        importance_cutoff: float,
        target_tokens: Optional[int],
        target_reduction: Optional[float],
    ) -> float:
        """Compute keep rate for ML compression."""
        if target_tokens is not None:
            rate = target_tokens / original_tokens if original_tokens > 0 else 1.0
            return max(0.1, min(0.95, rate))

        if target_reduction is not None:
            rate = 1.0 - target_reduction
            return max(0.1, min(0.95, rate))

        # Map importance_cutoff to rate
        # cutoff=0 -> rate=0.95 (minimal compression)
        # cutoff=0.5 -> rate=0.65
        # cutoff=1 -> rate=0.3 (aggressive compression)
        rate = 0.95 - (importance_cutoff * 0.65)
        return max(0.3, min(0.95, rate))


# =============================================================================
# Global Compressor Instances
# =============================================================================

_unified_compressor: Optional[UnifiedCompressor] = None
_hybrid_compressor: Optional[HybridCompressor] = None


def _get_unified_compressor(
    model_name: str = DEFAULT_ML_MODEL,
    device: str = "cpu",
    encoding: str = "o200k_base",
) -> UnifiedCompressor:
    global _unified_compressor
    if _unified_compressor is None:
        _unified_compressor = UnifiedCompressor(
            model_name=model_name,
            device=device,
            encoding=encoding,
        )
    return _unified_compressor


def _get_hybrid_compressor(encoding: str = "o200k_base") -> HybridCompressor:
    global _hybrid_compressor
    if _hybrid_compressor is None:
        _hybrid_compressor = HybridCompressor(encoding=encoding)
    return _hybrid_compressor


# =============================================================================
# Public API
# =============================================================================

def compress_text(
    prompt: str,
    importance_cutoff: float = 0.5,
    mode: str = "auto",
    target_tokens: Optional[int] = None,
    target_reduction: Optional[float] = None,
    force_tokens: Optional[List[str]] = None,
    device: str = "cpu",
    model_name: Optional[str] = None,
    use_embeddings: bool = True,
    query_text: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    encoding: str = "o200k_base",
    min_reduction: float = DEFAULT_MIN_REDUCTION,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    dedupe_level: str = "auto",
    max_context_tokens: Optional[int] = None,  # Alias for target_tokens
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compress text with given importance cutoff and mode.

    Args:
        prompt: Input text to compress
        importance_cutoff: 0-1, higher = more aggressive compression
        mode: Compression mode:
            - "auto": Use LLMLingua-2 if available, else hybrid (default)
            - "ml": Force LLMLingua-2 (falls back to hybrid if unavailable)
            - "hybrid": Use multi-stage hybrid compression
            - "identity": No compression
        target_tokens: Optional target token count
        target_reduction: Optional target reduction ratio (0-1)
        force_tokens: Tokens to preserve (ML mode only)
        device: "cpu" or "cuda" (ML mode only)
        model_name: LLMLingua model name (ML mode only)
        use_embeddings: Use embeddings for scoring
        query_text: Query for relevance scoring
        cache_dir: Cache directory
        use_cache: Whether to use caching
        encoding: Token encoding
        min_reduction: Minimum reduction target (default 0.05)
        near_dup_threshold: Similarity threshold for near-duplicate removal
        dedupe_level: "sentence", "paragraph", or "auto"
        max_context_tokens: Alias for target_tokens

    Returns:
        Tuple of (compressed_text, stats_dict)
    """
    mode = mode.lower()

    # Handle alias
    if max_context_tokens is not None and target_tokens is None:
        target_tokens = max_context_tokens

    # Validate mode
    valid_modes = {"auto", "ml", "hybrid", "identity"}
    if mode not in valid_modes:
        mode = "auto"

    # Identity mode
    if mode == "identity":
        tokens = count_tokens(prompt, encoding=encoding)
        return prompt, make_stats(tokens, tokens, "identity", stage_used=["identity"])

    # Check cache
    ml_model = model_name or DEFAULT_ML_MODEL
    if use_cache:
        cache = get_compression_cache(cache_dir)
        cached = cache.get(
            prompt, mode, importance_cutoff, target_tokens, target_reduction, ml_model
        )
        if cached:
            compressed, stats = cached
            stats["cached"] = True
            return compressed, stats

    # Compress based on mode
    if mode in ("auto", "ml"):
        # Use unified compressor (ML-first with hybrid fallback)
        compressor = _get_unified_compressor(ml_model, device, encoding)
        compressed, stats = compressor.compress(
            prompt,
            importance_cutoff=importance_cutoff,
            target_tokens=target_tokens,
            target_reduction=target_reduction,
            use_embeddings=use_embeddings,
            query_text=query_text,
            min_reduction=min_reduction,
            force_tokens=force_tokens,
            prefer_ml=(mode in ("auto", "ml")),
            near_dup_threshold=near_dup_threshold,
            dedupe_level=dedupe_level,
        )

    else:  # hybrid only
        compressor = _get_hybrid_compressor(encoding)
        compressed, stats = compressor.compress(
            prompt,
            importance_cutoff=importance_cutoff,
            target_tokens=target_tokens,
            target_reduction=target_reduction,
            use_embeddings=use_embeddings,
            cache_dir=cache_dir,
            query_text=query_text,
            min_reduction=min_reduction,
            near_dup_threshold=near_dup_threshold,
            dedupe_level=dedupe_level,
        )

    # Cache result
    if use_cache:
        cache = get_compression_cache(cache_dir)
        cache.set(
            prompt, mode, importance_cutoff, compressed, stats,
            target_tokens, target_reduction, ml_model
        )

    stats["cached"] = False
    return compressed, stats


def compress_messages(
    messages: List[Dict[str, str]],
    importance_cutoff: float = 0.5,
    mode: str = "auto",
    **kwargs,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Compress chat messages.

    System/developer messages and last user message are preserved.
    Other messages are compressed based on mode and cutoff.

    Args:
        messages: List of {"role": str, "content": str} messages
        importance_cutoff: 0-1, higher = more aggressive
        mode: Compression mode (auto, ml, hybrid, identity)
        **kwargs: Additional kwargs passed to compress_text

    Returns:
        Tuple of (compressed_messages, aggregate_stats)
    """
    if not messages:
        return messages, make_stats(0, 0, mode or "identity")

    encoding = kwargs.get("encoding", "o200k_base")

    # Find last user message
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"),
        default=-1
    )

    # Build query from system + last user
    system_text = "\n".join(
        m.get("content", "") for m in messages if m.get("role") in {"system", "developer"}
    )
    user_query = messages[last_user_idx].get("content", "") if last_user_idx >= 0 else ""
    query_text = f"{system_text}\n{user_query}".strip()

    total_original = 0
    total_compressed = 0
    compressed_messages = []
    methods = set()
    fallback_any = False
    errors = []
    stages = set()

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Protect system, developer, and last user
        is_protected = (
            role in {"system", "developer"} or
            (role == "user" and i == last_user_idx)
        )

        if is_protected or not content:
            compressed_messages.append(msg.copy())
            tokens = count_tokens(content, encoding=encoding)
            total_original += tokens
            total_compressed += tokens
            continue

        # Compress this message
        compressed_content, stats = compress_text(
            content,
            importance_cutoff=importance_cutoff,
            mode=mode,
            query_text=query_text,
            **kwargs,
        )

        new_msg = msg.copy()
        new_msg["content"] = compressed_content
        compressed_messages.append(new_msg)

        total_original += stats.get("original_tokens", 0)
        total_compressed += stats.get("compressed_tokens", 0)
        methods.add(stats.get("method", ""))
        fallback_any = fallback_any or stats.get("fallback_used", False)
        if stats.get("error"):
            errors.append(stats["error"])
        for s in stats.get("stage_used", []):
            stages.add(s)

    # Aggregate stats
    method = list(methods)[0] if len(methods) == 1 else "mixed"

    return compressed_messages, make_stats(
        total_original, total_compressed, method,
        fallback_used=fallback_any,
        error="; ".join(errors) if errors else "",
        messages_count=len(messages),
        stage_used=list(stages),
    )


def identity_compressor(
    prompt: str,
    importance_cutoff: float = 0.0,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Identity compressor - returns input unchanged (for baseline)."""
    encoding = kwargs.get("encoding", "o200k_base")
    tokens = count_tokens(prompt, encoding=encoding)
    return prompt, make_stats(tokens, tokens, "identity", stage_used=["identity"])


# =============================================================================
# Demo / Self-Test
# =============================================================================

def demo():
    """Demonstrate compression across modes."""
    short_prompt = """
    Please note that this is a test. What is 2+2?
    Answer with A/B/C/D.
    A. 3
    B. 4
    C. 5
    D. 6
    """

    # Prompt with redundancy for dedupe testing
    redundant_prompt = """
    The system is running normally. The system is running normally.
    All services are operational. All services are operational.
    No issues detected. Everything is working fine. No issues detected.

    Question: What is the status of the system?
    Answer with A/B/C/D.
    A. Running
    B. Stopped
    C. Error
    D. Unknown
    """

    long_prompt = """
    You are an expert assistant. Please help analyze the following data.

    Background Context:
    The system logs from 2024-01-15 show multiple transient failures across
    different services. It is important to note that the auth service experienced
    timeouts starting at 14:32:10. The incident ID is INC-20240115-12345.

    Additional Context:
    Various other systems were operating normally. The database cluster showed
    nominal performance metrics. Network latency was within expected bounds.
    Basically, most infrastructure was healthy during this period.

    Detailed Logs:
    ```
    2024-01-15 14:32:10 ERROR auth-service timeout after 30s
    2024-01-15 14:32:15 ERROR auth-service connection refused
    2024-01-15 14:32:20 INFO  auth-service recovered
    ```

    The investigation revealed that the root cause was a misconfigured timeout
    setting in the auth service configuration. The value was set to 30000ms
    instead of 30000.

    Question: What was the root cause of the auth service failures?
    Answer with A/B/C/D.

    A. Database connection pool exhaustion
    B. Misconfigured timeout setting
    C. Network connectivity issues
    D. Memory leak in the auth service
    """

    print("=" * 70)
    print("COMPRESSION DEMO (LLMLingua-2 + Hybrid Fallback)")
    print("=" * 70)
    print(f"\nLLMLingua-2 available: {LLMLINGUA_AVAILABLE}")
    print(f"Sentence-transformers available: {SentenceTransformer is not None}")
    print(f"BM25 available: {BM25Okapi is not None}")

    # Test each mode
    for mode in ["auto", "hybrid", "identity"]:
        print(f"\n--- Mode: {mode.upper()} ---")

        # Short prompt
        compressed, stats = compress_text(
            short_prompt,
            importance_cutoff=0.5,
            mode=mode,
            use_cache=False,
        )
        print(f"Short prompt: {stats['original_tokens']} -> {stats['compressed_tokens']} "
              f"({stats['reduction_ratio']:.1%} reduction)")
        print(f"  Method: {stats.get('method', 'unknown')}, Stages: {stats.get('stage_used', [])}")
        if stats.get("fallback_used"):
            print(f"  Fallback: {stats.get('error', 'unknown')}")

        # Redundant prompt
        compressed, stats = compress_text(
            redundant_prompt,
            importance_cutoff=0.5,
            mode=mode,
            use_cache=False,
        )
        print(f"Redundant prompt: {stats['original_tokens']} -> {stats['compressed_tokens']} "
              f"({stats['reduction_ratio']:.1%} reduction)")
        print(f"  Method: {stats.get('method', 'unknown')}, dedupe_removed: {stats.get('dedupe_removed', 0)}")

        # Long prompt
        compressed, stats = compress_text(
            long_prompt,
            importance_cutoff=0.7,
            mode=mode,
            use_cache=False,
        )
        print(f"Long prompt: {stats['original_tokens']} -> {stats['compressed_tokens']} "
              f"({stats['reduction_ratio']:.1%} reduction)")
        print(f"  Method: {stats.get('method', 'unknown')}, Stages: {stats.get('stage_used', [])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
