"""
Prompt Compressor Module

A deterministic prompt compression system that reduces token count while preserving
critical information. Provides a clean public API for integration with evaluation systems.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass

# Optional dependencies
TIKTOKEN_AVAILABLE = False
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    pass


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
        punct_count = len(re.findall(r'[.,!?;:"\'()\[\]{}]', text))
        return int((len(words) + punct_count) * 1.3)


# Global token counter instance
_token_counter = None

def get_token_counter(encoding: str = "o200k_base") -> TokenCounter:
    """Get or create token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter(encoding)
    return _token_counter


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Count tokens in text."""
    return get_token_counter(encoding).count(text)


@dataclass
class CompressionStats:
    """Statistics from compression operation."""
    original_tokens: int
    compressed_tokens: int
    reduction_ratio: float
    method: str = "heuristic"


class SimpleCompressor:
    """
    Simple heuristic-based compressor for prompt text.

    Compression strategies:
    1. Remove redundant whitespace
    2. Truncate very long paragraphs
    3. Remove filler phrases
    4. Prioritize content near the end (recency bias)
    """

    # Filler phrases that can be removed
    FILLER_PHRASES = [
        r'\bplease note that\b',
        r'\bit is important to note that\b',
        r'\bas mentioned earlier\b',
        r'\bas we discussed\b',
        r'\bin other words\b',
        r'\bthat being said\b',
        r'\bhaving said that\b',
        r'\bwith that in mind\b',
        r'\bfor your information\b',
        r'\bfor reference\b',
        r'\bbasically\b',
        r'\bessentially\b',
        r'\bactually\b',
        r'\bobviously\b',
        r'\bclearly\b',
        r'\bsimply put\b',
        r'\bin summary\b',
        r'\bto summarize\b',
        r'\bin conclusion\b',
        r'\ball in all\b',
    ]

    # Content that should never be removed
    PROTECTED_PATTERNS = [
        r'```[\s\S]*?```',  # Code blocks
        r'\{[^{}]*\}',  # JSON-like
        r'"[^"]*"',  # Quoted strings
        r"'[^']*'",  # Single quoted
        r'\b\d+\.\d+\b',  # Decimals
        r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',  # Dates
        r'https?://[^\s]+',  # URLs
    ]

    def __init__(self, encoding: str = "o200k_base"):
        self.token_counter = TokenCounter(encoding)
        self.filler_regex = re.compile(
            '|'.join(self.FILLER_PHRASES),
            re.IGNORECASE
        )

    def _extract_protected(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract protected content and replace with placeholders."""
        placeholders = {}
        result = text

        for i, pattern in enumerate(self.PROTECTED_PATTERNS):
            for j, match in enumerate(re.finditer(pattern, result)):
                key = f"__PROTECTED_{i}_{j}__"
                placeholders[key] = match.group()
                result = result.replace(match.group(), key, 1)

        return result, placeholders

    def _restore_protected(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore protected content from placeholders."""
        result = text
        for key, value in placeholders.items():
            result = result.replace(key, value)
        return result

    def _remove_fillers(self, text: str) -> str:
        """Remove filler phrases."""
        return self.filler_regex.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Multiple spaces to single
        text = re.sub(r' +', ' ', text)
        # Multiple newlines to double
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _truncate_by_importance(
        self,
        text: str,
        target_tokens: int,
        importance_cutoff: float
    ) -> str:
        """Truncate text based on importance cutoff."""
        paragraphs = text.split('\n\n')
        if len(paragraphs) <= 2:
            return text

        # Calculate how many paragraphs to keep
        # Lower cutoff = keep more, higher cutoff = keep less
        keep_ratio = 1.0 - importance_cutoff
        keep_count = max(2, int(len(paragraphs) * keep_ratio))

        # Prioritize first and last paragraphs
        if keep_count >= len(paragraphs):
            return text

        # Keep first paragraph, last paragraphs, and sample from middle
        first = paragraphs[0]
        last_count = min(keep_count - 1, len(paragraphs) - 1)
        last_paras = paragraphs[-last_count:] if last_count > 0 else []

        kept = [first] + last_paras
        return '\n\n'.join(kept)

    def compress(
        self,
        text: str,
        importance_cutoff: float = 0.5,
        target_tokens: Optional[int] = None
    ) -> Tuple[str, CompressionStats]:
        """
        Compress text using heuristic methods.

        Args:
            text: Input text to compress
            importance_cutoff: Float 0-1, higher = more aggressive compression
            target_tokens: Optional target token count

        Returns:
            Tuple of (compressed_text, stats)
        """
        if not text:
            return text, CompressionStats(0, 0, 0.0)

        original_tokens = self.token_counter.count(text)

        # Extract protected content
        working_text, placeholders = self._extract_protected(text)

        # Apply compression steps
        working_text = self._remove_fillers(working_text)
        working_text = self._normalize_whitespace(working_text)

        # If target specified or cutoff high, truncate
        if target_tokens or importance_cutoff > 0.3:
            target = target_tokens or int(original_tokens * (1 - importance_cutoff * 0.5))
            working_text = self._truncate_by_importance(
                working_text, target, importance_cutoff
            )

        # Restore protected content
        compressed = self._restore_protected(working_text, placeholders)
        compressed = self._normalize_whitespace(compressed)

        compressed_tokens = self.token_counter.count(compressed)
        reduction = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0.0

        return compressed, CompressionStats(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            reduction_ratio=reduction,
            method="heuristic"
        )


# Module-level compressor instance
_compressor = None

def get_compressor() -> SimpleCompressor:
    """Get or create compressor instance."""
    global _compressor
    if _compressor is None:
        _compressor = SimpleCompressor()
    return _compressor


def compress_text(
    prompt: str,
    importance_cutoff: float = 0.5,
    **kwargs
) -> Tuple[str, dict]:
    """
    Public API: Compress text with given importance cutoff.

    Args:
        prompt: Input text to compress
        importance_cutoff: Float 0-1, higher = more aggressive compression
        **kwargs: Additional options (target_tokens, encoding)

    Returns:
        Tuple of (compressed_text, stats_dict)
    """
    target_tokens = kwargs.get('target_tokens')
    compressor = get_compressor()

    compressed, stats = compressor.compress(
        prompt,
        importance_cutoff=importance_cutoff,
        target_tokens=target_tokens
    )

    return compressed, {
        'original_tokens': stats.original_tokens,
        'compressed_tokens': stats.compressed_tokens,
        'reduction_ratio': stats.reduction_ratio,
        'method': stats.method,
    }


def compress_messages(
    messages: List[Dict[str, str]],
    importance_cutoff: float = 0.5,
    **kwargs
) -> Tuple[List[Dict[str, str]], dict]:
    """
    Public API: Compress chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        importance_cutoff: Float 0-1, higher = more aggressive compression
        **kwargs: Additional options

    Returns:
        Tuple of (compressed_messages, stats_dict)
    """
    if not messages:
        return messages, {'original_tokens': 0, 'compressed_tokens': 0, 'reduction_ratio': 0.0}

    compressor = get_compressor()
    total_original = 0
    total_compressed = 0

    compressed_messages = []

    for i, msg in enumerate(messages):
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Protect system messages and last user message
        is_protected = (
            role == 'system' or
            role == 'developer' or
            (role == 'user' and i == len(messages) - 1)
        )

        if is_protected or not content:
            compressed_messages.append(msg.copy())
            total_original += count_tokens(content)
            total_compressed += count_tokens(content)
        else:
            compressed_content, stats = compressor.compress(
                content, importance_cutoff=importance_cutoff
            )
            new_msg = msg.copy()
            new_msg['content'] = compressed_content
            compressed_messages.append(new_msg)
            total_original += stats.original_tokens
            total_compressed += stats.compressed_tokens

    reduction = (total_original - total_compressed) / total_original if total_original > 0 else 0.0

    return compressed_messages, {
        'original_tokens': total_original,
        'compressed_tokens': total_compressed,
        'reduction_ratio': reduction,
        'messages_count': len(messages),
    }


# Identity compressor for baseline
def identity_compressor(prompt: str, importance_cutoff: float = 0.0, **kwargs) -> Tuple[str, dict]:
    """Identity compressor - returns input unchanged (for baseline)."""
    tokens = count_tokens(prompt)
    return prompt, {
        'original_tokens': tokens,
        'compressed_tokens': tokens,
        'reduction_ratio': 0.0,
        'method': 'identity',
    }


if __name__ == "__main__":
    # Quick test
    test_text = """
    Please note that this is a test of the compression system. It is important to note that
    we want to preserve critical information while removing filler content.

    Here is some code that must be preserved:
    ```python
    def hello():
        return "world"
    ```

    The date is 2024-01-15 and the value is 3.14159.

    In conclusion, this text should be shorter after compression.
    """

    compressed, stats = compress_text(test_text, importance_cutoff=0.5)
    print(f"Original: {stats['original_tokens']} tokens")
    print(f"Compressed: {stats['compressed_tokens']} tokens")
    print(f"Reduction: {stats['reduction_ratio']:.1%}")
    print(f"\nCompressed text:\n{compressed}")
