"""
Evals package for LongBench v2 evaluation.

Modules:
    - compressor: Text compression with public API
    - longbench_eval: Evaluation module with CLI
"""

from .compressor import compress_text, compress_messages, count_tokens, identity_compressor
from .longbench_eval import run_experiment

__all__ = [
    'compress_text',
    'compress_messages',
    'count_tokens',
    'identity_compressor',
    'run_experiment',
]
