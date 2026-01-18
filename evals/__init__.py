"""
Evals package for LongBench v2 evaluation.

Modules:
    - compressor: Multi-mode text compression (ML, hybrid, heuristic, identity)
    - longbench_eval: Evaluation module with CLI

Compression Modes:
    - auto: Use LLMLingua-2 if available, else hybrid (default)
    - ml: Force LLMLingua-2 (falls back to hybrid if unavailable)
    - hybrid: Multi-stage (minify + dedupe + chunk selection)
    - identity: No compression (baseline)

Usage:
    from evals import compress_text, compress_messages

    # Auto mode (ML-first with hybrid fallback)
    compressed, stats = compress_text(prompt, importance_cutoff=0.5, mode="auto")

    # Hybrid compression only
    compressed, stats = compress_text(prompt, importance_cutoff=0.5, mode="hybrid")

    # Message compression
    compressed_msgs, stats = compress_messages(messages, mode="auto")
"""

from .compressor import (
    compress_text,
    compress_messages,
    identity_compressor,
    count_tokens,
    # Additional exports for advanced usage
    LLMLingua2Compressor,
    HybridCompressor,
    UnifiedCompressor,
    LLMLINGUA_AVAILABLE,
)

__all__ = [
    # Core API
    'compress_text',
    'compress_messages',
    'identity_compressor',
    'count_tokens',
    # Compressor classes
    'LLMLingua2Compressor',
    'HybridCompressor',
    'UnifiedCompressor',
    # Availability flags
    'LLMLINGUA_AVAILABLE',
]
