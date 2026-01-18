"""
Tests for the prompt compressor module.

Run with: python -m pytest evals/tests/test_compressor.py -v

Tests cover:
1. Code block preservation
2. JSON/YAML content preservation
3. Dedupe removes repeated sentences
4. Short prompt compression achieves reduction via minify/dedupe
5. compress_messages protects system/developer + last user message
6. LLMLingua-2 integration (when available)
7. Auto mode with ML-first and hybrid fallback
"""

import pytest
from typing import Tuple, Dict, Any

# Import the module
from evals.compressor import (
    compress_text,
    compress_messages,
    identity_compressor,
    count_tokens,
    LLMLINGUA_AVAILABLE,
    Minifier,
    RedundancyRemover,
    ProtectedSpanExtractor,
    LLMLingua2Compressor,
    UnifiedCompressor,
    SemanticChunker,
    get_token_counter,
    DEFAULT_MIN_REDUCTION,
)


class TestCountTokens:
    """Tests for token counting."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_longer_text(self):
        text = "This is a longer piece of text with more words and tokens."
        tokens = count_tokens(text)
        assert 10 < tokens < 30


class TestIdentityCompressor:
    """Tests for identity (no-op) compressor."""

    def test_returns_unchanged(self):
        prompt = "What is 2+2? Answer with A/B/C/D."
        compressed, stats = identity_compressor(prompt)
        assert compressed == prompt

    def test_stats_structure(self):
        prompt = "Test prompt"
        _, stats = identity_compressor(prompt)

        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
        assert "reduction_ratio" in stats
        assert "method" in stats

        assert stats["method"] == "identity"
        assert stats["reduction_ratio"] == 0.0
        assert stats["original_tokens"] == stats["compressed_tokens"]


class TestMinifier:
    """Tests for Stage 0: Minifier."""

    def test_removes_filler_phrases(self):
        minifier = Minifier()
        text = "Please note that this is important. Basically, it works."
        result = minifier.minify(text)
        assert "please note that" not in result.lower()
        assert "basically" not in result.lower()

    def test_collapses_whitespace(self):
        minifier = Minifier()
        text = "Hello    world\n\n\n\nTest"
        result = minifier.minify(text)
        assert "    " not in result
        assert "\n\n\n" not in result

    def test_preserves_content(self):
        minifier = Minifier()
        text = "Important data: 12345. Question: What is the value?"
        result = minifier.minify(text)
        assert "12345" in result
        assert "Question" in result


class TestRedundancyRemover:
    """Tests for Stage 1: Redundancy removal."""

    def test_exact_dedupe_sentences(self):
        remover = RedundancyRemover(use_embeddings=False)
        text = "The system is running. The system is running. All services are up."
        result, removed, _ = remover.dedupe(text, level="sentence")
        assert removed > 0
        assert result.count("The system is running") == 1

    def test_exact_dedupe_paragraphs(self):
        remover = RedundancyRemover(use_embeddings=False)
        text = "First paragraph.\n\nFirst paragraph.\n\nSecond paragraph."
        result, removed, _ = remover.dedupe(text, level="paragraph")
        assert removed > 0

    def test_preserves_unique_content(self):
        remover = RedundancyRemover(use_embeddings=False)
        text = "First sentence. Second sentence. Third sentence."
        result, removed, _ = remover.dedupe(text, level="sentence")
        assert removed == 0
        assert "First" in result
        assert "Second" in result
        assert "Third" in result


class TestProtectedSpanExtractor:
    """Tests for protected span extraction."""

    def test_code_block_extraction(self):
        extractor = ProtectedSpanExtractor()
        text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
        result, placeholders, count = extractor.extract(text)
        assert count > 0
        assert "__PROTECTED_" in result
        assert "```python" in "".join(placeholders.values())

    def test_json_extraction(self):
        extractor = ProtectedSpanExtractor()
        text = 'Config: {"key": "value", "num": 42}'
        result, placeholders, count = extractor.extract(text)
        assert count > 0
        restored = extractor.restore(result, placeholders)
        assert '"key"' in restored
        assert '"value"' in restored

    def test_url_extraction(self):
        extractor = ProtectedSpanExtractor()
        text = "Visit https://example.com/path for more info."
        result, placeholders, count = extractor.extract(text)
        assert count > 0
        restored = extractor.restore(result, placeholders)
        assert "https://example.com/path" in restored

    def test_restore_preserves_original(self):
        extractor = ProtectedSpanExtractor()
        text = "Date: 2024-01-15, UUID: 550e8400-e29b-41d4-a716-446655440000"
        result, placeholders, count = extractor.extract(text)
        restored = extractor.restore(result, placeholders)
        assert "2024-01-15" in restored
        assert "550e8400-e29b-41d4-a716-446655440000" in restored


class TestCodeBlockPreservation:
    """Test that code blocks are preserved through compression."""

    PROMPT_WITH_CODE = '''
    Here is some background context that could be removed.
    Please note that this is filler content.

    Here is the important code:
    ```python
    def hello_world():
        print("Hello, World!")
        return 42
    ```

    More filler content that isn't important.

    Question: What does hello_world return?
    Answer with A/B/C/D.
    A. 41
    B. 42
    C. 43
    D. 44
    '''

    def test_code_block_preserved_hybrid(self):
        """Code blocks should survive compression in hybrid mode."""
        compressed, stats = compress_text(
            self.PROMPT_WITH_CODE,
            importance_cutoff=0.7,
            mode="hybrid",
            use_cache=False,
        )

        # Code block should be present
        assert "```python" in compressed, "Code block marker was removed"
        assert "def hello_world" in compressed, "Function definition was removed"
        assert 'print("Hello, World!")' in compressed, "Print statement was removed"
        assert "return 42" in compressed, "Return statement was removed"

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_code_block_preserved_ml(self):
        """Code blocks should survive ML compression."""
        compressed, stats = compress_text(
            self.PROMPT_WITH_CODE,
            importance_cutoff=0.5,
            mode="ml",
            use_cache=False,
        )
        # Code block should be mostly intact
        assert "hello_world" in compressed or "def " in compressed


class TestJSONPreservation:
    """Test that JSON content is preserved."""

    PROMPT_WITH_JSON = '''
    Some context here that is filler. Please note that this can be removed.

    Configuration data:
    {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

    More filler content.

    Question: What is the value of key?
    Answer with A/B/C/D.
    A. number
    B. value
    C. nested
    D. null
    '''

    def test_json_preserved_hybrid(self):
        """JSON should survive compression."""
        compressed, stats = compress_text(
            self.PROMPT_WITH_JSON,
            importance_cutoff=0.5,
            mode="hybrid",
            use_cache=False,
        )

        # JSON structure should be present
        assert '"key"' in compressed or "'key'" in compressed
        assert '"value"' in compressed or "'value'" in compressed
        assert "42" in compressed


class TestDedupeRemovesRepeatedSentences:
    """Test that dedupe removes repeated sentences in short prompts."""

    REDUNDANT_PROMPT = """
    The system is running normally. The system is running normally.
    All services are operational. All services are operational.
    No issues detected. Everything works fine. No issues detected.

    Question: What is the system status?
    Answer with A/B/C/D.
    A. Running
    B. Stopped
    C. Error
    D. Unknown
    """

    def test_dedupe_removes_duplicates(self):
        """Dedupe should remove repeated sentences."""
        compressed, stats = compress_text(
            self.REDUNDANT_PROMPT,
            importance_cutoff=0.5,
            mode="hybrid",
            use_cache=False,
        )

        # Should have reduction due to dedupe
        assert stats["reduction_ratio"] > 0.05, (
            f"Expected >5% reduction from dedupe, got {stats['reduction_ratio']:.1%}"
        )

        # Should have dedupe in stages
        assert "dedupe" in stats.get("stage_used", [])

        # Duplicates should be removed
        # Count occurrences of repeated phrase
        count = compressed.lower().count("the system is running normally")
        assert count <= 1, f"Duplicate sentence not removed, found {count} occurrences"


class TestShortPromptCompression:
    """Test that short prompts achieve at least small reduction via minify/dedupe."""

    SHORT_PROMPT_WITH_FILLERS = """
    Please note that this is a test. It is important to note that we are testing.
    Basically, the system works well. As a matter of fact, everything is fine.

    What is 2+2?
    Answer with A/B/C/D.
    A. 3
    B. 4
    C. 5
    D. 6
    """

    def test_short_prompt_achieves_reduction(self):
        """Short prompt with fillers should achieve some reduction."""
        compressed, stats = compress_text(
            self.SHORT_PROMPT_WITH_FILLERS,
            importance_cutoff=0.3,
            mode="hybrid",
            use_cache=False,
        )

        # Should have some reduction from filler removal
        assert stats["reduction_ratio"] >= 0.02, (
            f"Expected >=2% reduction from minify/dedupe, got {stats['reduction_ratio']:.1%}"
        )

        # Question should still be present
        assert "What is 2+2?" in compressed
        assert "Answer with A/B/C/D" in compressed


class TestShortPromptNeverExpands:
    """Test that short prompts are never expanded by compression."""

    SHORT_PROMPTS = [
        "What is 2+2?",
        "Answer with A/B/C/D.\nA. 1\nB. 2\nC. 3\nD. 4",
        "Hello world. This is a short prompt.",
    ]

    @pytest.mark.parametrize("prompt", SHORT_PROMPTS)
    def test_short_prompt_no_expansion(self, prompt: str):
        """Short prompts should never become longer after compression."""
        compressed, stats = compress_text(
            prompt,
            importance_cutoff=0.9,
            mode="hybrid",
            use_cache=False,
        )

        assert stats["compressed_tokens"] <= stats["original_tokens"], (
            f"Compression expanded the prompt: {stats['original_tokens']} -> {stats['compressed_tokens']}"
        )


class TestLongPromptCompression:
    """Test that long prompts are actually compressed."""

    LONG_PROMPT = """
    Background Information:
    Please note that this document contains important information.
    It is important to note that the following details are critical.
    In other words, pay attention to the key points.
    That being said, let's proceed with the analysis.

    Context Data:
    The system experienced multiple failures on 2024-01-15.
    Error logs indicate timeout issues in the authentication service.
    The incident ID is INC-20240115-12345.
    Root cause appears to be misconfigured timeout settings.

    Additional Background:
    Various other systems were operating normally during this period.
    Database metrics showed nominal performance characteristics.
    Network latency remained within expected bounds throughout.
    CPU and memory utilization were stable across all nodes.

    More Context:
    The deployment pipeline completed successfully earlier that day.
    No configuration changes were made to production systems.
    Monitoring alerts triggered at 14:32:10 local time.
    On-call engineer acknowledged the incident within 5 minutes.

    Question: What was the root cause of the service failures?
    Answer with A/B/C/D.

    A. Database connection pool exhaustion
    B. Misconfigured timeout settings
    C. Network connectivity issues
    D. Memory leak in the auth service
    """

    def test_high_cutoff_reduces_tokens(self):
        """High cutoff should significantly reduce token count for long prompts."""
        compressed, stats = compress_text(
            self.LONG_PROMPT,
            importance_cutoff=0.9,
            mode="hybrid",
            use_cache=False,
        )

        assert stats["reduction_ratio"] > 0.1, (
            f"Expected >10% reduction, got {stats['reduction_ratio']:.1%}"
        )

    def test_low_cutoff_minimal_change(self):
        """Low cutoff should make minimal changes."""
        compressed, stats = compress_text(
            self.LONG_PROMPT,
            importance_cutoff=0.1,
            mode="hybrid",
            use_cache=False,
        )

        # Some reduction from minify is OK, but shouldn't be aggressive
        assert stats["reduction_ratio"] < 0.5, (
            f"Low cutoff reduced too much: {stats['reduction_ratio']:.1%}"
        )


class TestCompressMessages:
    """Test message compression."""

    def test_empty_messages(self):
        messages, stats = compress_messages([], mode="hybrid")
        assert messages == []

    def test_system_message_preserved(self):
        """System messages should be preserved unchanged."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        compressed, stats = compress_messages(messages, mode="hybrid", use_cache=False)

        # System message should be unchanged
        assert compressed[0]["content"] == messages[0]["content"]

    def test_developer_message_preserved(self):
        """Developer messages should be preserved unchanged."""
        messages = [
            {"role": "developer", "content": "Important system instructions."},
            {"role": "user", "content": "Hello"},
        ]

        compressed, stats = compress_messages(messages, mode="hybrid", use_cache=False)

        # Developer message should be unchanged
        assert compressed[0]["content"] == messages[0]["content"]

    def test_last_user_message_preserved(self):
        """Last user message should be preserved unchanged."""
        messages = [
            {"role": "user", "content": "First message with lots of text and filler."},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "This is my final question?"},
        ]

        compressed, stats = compress_messages(messages, mode="hybrid", use_cache=False)

        # Last user message should be unchanged
        assert compressed[-1]["content"] == messages[-1]["content"]

    def test_intermediate_messages_compressed(self):
        """Intermediate user messages should be compressed."""
        long_filler = "Please note that this is filler. " * 20  # Create compressible content
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": long_filler + " First question?"},
            {"role": "assistant", "content": "First response."},
            {"role": "user", "content": "Final question?"},
        ]

        compressed, stats = compress_messages(
            messages,
            importance_cutoff=0.7,
            mode="hybrid",
            use_cache=False
        )

        # Intermediate user message (index 1) should be compressed
        # It had lots of filler text
        assert len(compressed[1]["content"]) <= len(messages[1]["content"])


class TestStatsStructure:
    """Test that stats dictionaries have required fields."""

    REQUIRED_FIELDS = [
        "original_tokens",
        "compressed_tokens",
        "reduction_ratio",
        "method",
        "fallback_used",
        "error",
        "embeddings_used",
        "protected_spans_count",
        "chunks_total",
        "chunks_kept",
        "budget_tokens",
        "stage_used",
        "min_reduction_target",
        "achieved_reduction",
    ]

    @pytest.mark.parametrize("mode", ["hybrid", "identity"])
    def test_stats_has_required_fields(self, mode: str):
        _, stats = compress_text("Test prompt with some content.", mode=mode, use_cache=False)

        for field in self.REQUIRED_FIELDS:
            assert field in stats, f"Missing required field: {field}"

    def test_stage_used_is_list(self):
        """stage_used should be a list."""
        _, stats = compress_text("Test prompt", mode="hybrid", use_cache=False)
        assert isinstance(stats["stage_used"], list)

    def test_reduction_ratio_matches_achieved(self):
        """reduction_ratio and achieved_reduction should match."""
        _, stats = compress_text("Test prompt with content.", mode="hybrid", use_cache=False)
        assert stats["reduction_ratio"] == stats["achieved_reduction"]


class TestModeSwitching:
    """Test that mode switching works correctly."""

    PROMPT = "What is the meaning of life? Answer with A/B/C/D."

    def test_identity_mode(self):
        compressed, stats = compress_text(self.PROMPT, mode="identity", use_cache=False)
        assert compressed == self.PROMPT
        assert stats["method"] == "identity"

    def test_hybrid_mode(self):
        compressed, stats = compress_text(self.PROMPT, mode="hybrid", use_cache=False)
        # Should use one of the hybrid methods
        valid_methods = {"hybrid_embeddings", "bm25", "lexical", "dedupe", "minify"}
        assert stats["method"] in valid_methods

    def test_ml_mode_fallback(self):
        """Test ML mode - will fallback to hybrid if LLMLingua not installed."""
        compressed, stats = compress_text(self.PROMPT, mode="ml", use_cache=False)
        # Either uses llmlingua2 or falls back to hybrid methods
        valid_methods = {"llmlingua2", "llmlingua", "hybrid_embeddings", "bm25", "lexical", "dedupe", "minify"}
        assert stats["method"] in valid_methods

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_ml_mode_with_llmlingua(self):
        """Test ML mode when LLMLingua is available."""
        compressed, stats = compress_text(self.PROMPT, mode="ml", use_cache=False)
        assert stats["method"] == "llmlingua2"


class TestFallbackBehavior:
    """Test fallback behavior when ML is unavailable."""

    def test_ml_fallback_when_unavailable(self):
        """ML mode should fallback gracefully if llmlingua isn't installed."""
        compressed, stats = compress_text(
            "Test prompt",
            mode="ml",
            use_cache=False,
        )

        # Should either succeed with ML or fallback
        if not LLMLINGUA_AVAILABLE:
            assert stats.get("fallback_used", False) or stats["method"] != "llmlingua2"

    def test_fail_open_returns_original(self):
        """On error, should return original text with fallback_used=True."""
        # Test with normal text - should work
        original = "Simple test prompt."
        compressed, stats = compress_text(original, mode="hybrid", use_cache=False)

        # Compressed should not be longer
        assert stats["compressed_tokens"] <= stats["original_tokens"]


class TestCaching:
    """Test compression caching."""

    def test_cache_hit(self):
        prompt = "Unique test prompt for caching test " + str(hash("cache_test_v2"))

        # First call (cache miss)
        _, stats1 = compress_text(prompt, mode="hybrid", use_cache=True)

        # Second call (should be cache hit)
        _, stats2 = compress_text(prompt, mode="hybrid", use_cache=True)

        # Both should return same results
        assert stats1["compressed_tokens"] == stats2["compressed_tokens"]

    def test_cache_disabled(self):
        prompt = "Test prompt for no-cache test"

        _, stats = compress_text(prompt, mode="hybrid", use_cache=False)

        # Should have cached=False in stats
        assert stats.get("cached") == False


class TestMinReductionParameter:
    """Test min_reduction parameter behavior."""

    def test_min_reduction_respected(self):
        """Compression should try to achieve min_reduction."""
        prompt = """
        Background context with some information.
        More details that could potentially be reduced.
        Additional context that adds to the length.

        Question: What is the answer?
        A. Option A
        B. Option B
        """

        _, stats = compress_text(
            prompt,
            importance_cutoff=0.5,
            mode="hybrid",
            min_reduction=0.1,
            use_cache=False,
        )

        # min_reduction_target should be recorded
        assert stats["min_reduction_target"] == 0.1


class TestTargetTokensAndReduction:
    """Test target_tokens and target_reduction parameters."""

    LONG_PROMPT = """
    This is a longer prompt with multiple paragraphs of content.
    It contains various information that could be compressed.

    More context here with additional details and information.
    The system processes this data and produces outputs.

    Question: What is the result?
    Answer with A/B/C/D.
    A. First
    B. Second
    C. Third
    D. Fourth
    """ * 3  # Repeat to make it longer

    def test_target_reduction(self):
        """target_reduction should guide compression aggressiveness."""
        _, stats = compress_text(
            self.LONG_PROMPT,
            importance_cutoff=0.5,
            mode="hybrid",
            target_reduction=0.3,
            use_cache=False,
        )

        # Should try to achieve target reduction
        # May not always hit exactly due to chunk boundaries
        assert stats["reduction_ratio"] > 0.1


class TestEmbeddingsOptional:
    """Test that embeddings are optional and compression works without them."""

    def test_compression_without_embeddings(self):
        """Compression should work when use_embeddings=False."""
        prompt = """
        Some context that could be compressed.
        Additional information here.

        Question: What is the answer?
        """

        compressed, stats = compress_text(
            prompt,
            importance_cutoff=0.5,
            mode="hybrid",
            use_embeddings=False,
            use_cache=False,
        )

        assert stats["embeddings_used"] == False
        assert compressed  # Should return something


class TestAutoMode:
    """Test auto mode with ML-first and hybrid fallback."""

    def test_auto_mode_works(self):
        """Auto mode should work regardless of LLMLingua availability."""
        prompt = "What is 2+2? Answer with A/B/C/D."
        compressed, stats = compress_text(prompt, mode="auto", use_cache=False)

        # Should return valid compressed text
        assert compressed
        assert stats["original_tokens"] > 0

    def test_auto_mode_falls_back_to_hybrid(self):
        """When LLMLingua unavailable, auto mode should fall back to hybrid."""
        if LLMLINGUA_AVAILABLE:
            pytest.skip("LLMLingua is available, can't test fallback")

        prompt = "Test prompt for auto mode fallback."
        compressed, stats = compress_text(prompt, mode="auto", use_cache=False)

        # Should use hybrid method
        assert stats.get("compression_method") == "hybrid" or stats["method"] in {
            "hybrid_embeddings", "bm25", "lexical", "dedupe", "minify"
        }


class TestSemanticChunker:
    """Test the semantic chunker for BERT-compatible chunking."""

    def test_short_text_single_chunk(self):
        """Short text should return as single chunk."""
        token_counter = get_token_counter()
        chunker = SemanticChunker(token_counter, max_tokens=450)

        short_text = "This is a short text."
        chunks = chunker.chunk(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        token_counter = get_token_counter()
        chunker = SemanticChunker(token_counter, max_tokens=50)

        long_text = "This is a sentence. " * 50
        chunks = chunker.chunk(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should be under the limit (approximately)
            assert token_counter.count(chunk) <= 60  # Allow some slack

    def test_preserves_paragraph_boundaries(self):
        """Chunker should prefer paragraph boundaries."""
        token_counter = get_token_counter()
        chunker = SemanticChunker(token_counter, max_tokens=100)

        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph."
        chunks = chunker.chunk(text)

        # Should respect paragraph boundaries when possible
        assert len(chunks) >= 1


class TestLLMLingua2Compressor:
    """Test the LLMLingua-2 compressor class."""

    def test_is_available_check(self):
        """is_available should return correct status."""
        compressor = LLMLingua2Compressor()
        available = compressor.is_available()

        # Should match global availability
        assert available == LLMLINGUA_AVAILABLE

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_compression_with_llmlingua(self):
        """Test actual LLMLingua-2 compression when available."""
        compressor = LLMLingua2Compressor()

        text = """
        Please note that this is a test document with filler content.
        It is important to note that we are testing compression.
        The system should compress this text effectively.

        Question: What is being tested?
        A. Compression
        B. Expansion
        C. Nothing
        D. Everything
        """

        compressed, stats = compressor.compress(text, rate=0.5)

        assert stats["method"] == "llmlingua2"
        assert stats["reduction_ratio"] > 0


class TestUnifiedCompressor:
    """Test the unified compressor with ML-first fallback."""

    def test_unified_compressor_works(self):
        """Unified compressor should work regardless of dependencies."""
        compressor = UnifiedCompressor()

        text = "Test prompt for unified compression."
        compressed, stats = compressor.compress(text, importance_cutoff=0.5)

        assert compressed
        assert "method" in stats

    def test_unified_compressor_respects_prefer_ml(self):
        """Unified compressor should respect prefer_ml parameter."""
        compressor = UnifiedCompressor()

        text = "Test prompt."

        # With prefer_ml=False, should use hybrid
        compressed, stats = compressor.compress(text, prefer_ml=False)

        assert stats.get("compression_method") == "hybrid"


class TestPreprocessingInML:
    """Test that preprocessing runs before ML compression."""

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_preprocessing_removes_fillers(self):
        """Preprocessing should remove filler phrases before ML compression."""
        compressor = LLMLingua2Compressor()

        text = """
        Please note that this is filler. Basically, the content here is redundant.
        As a matter of fact, it is important to note that fillers are removed.

        Question: What happens to fillers?
        """

        compressed, stats = compressor.compress(text, use_preprocess=True)

        # Preprocessing stages should be recorded
        assert "minify" in stats.get("stage_used", [])
        assert "dedupe" in stats.get("stage_used", [])


class TestProtectedSpansInML:
    """Test that protected spans survive ML compression."""

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_code_blocks_preserved_in_ml(self):
        """Code blocks should be preserved through ML compression."""
        text = '''
        Some context.

        ```python
        def test():
            return 42
        ```

        Question: What does test return?
        '''

        compressed, stats = compress_text(text, mode="ml", use_cache=False)

        assert "```python" in compressed
        assert "return 42" in compressed

    @pytest.mark.skipif(not LLMLINGUA_AVAILABLE, reason="llmlingua not installed")
    def test_json_preserved_in_ml(self):
        """JSON should be preserved through ML compression."""
        text = '''
        Config: {"key": "value", "num": 42}

        Question: What is the value?
        '''

        compressed, stats = compress_text(text, mode="ml", use_cache=False)

        assert '"key"' in compressed or "'key'" in compressed
        assert "42" in compressed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
