# LLM Input Compressor & LongBench v2 Evaluation

A deterministic prompt compression system with a mini LongBench v2 evaluation framework for benchmarking compression quality on OpenRouter.

## Overview

This project provides:

1. **Prompt Compressor** - Reduces token count while preserving critical information
2. **LongBench v2 Evaluator** - Benchmarks compression impact on LLM accuracy using the [LongBench v2 dataset](https://huggingface.co/datasets/zai-org/LongBench-v2)

The evaluation runs multiple compression conditions (baseline, various cutoff levels) and measures accuracy degradation vs token savings, all within a configurable budget.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd polar-bear-1

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Basic Usage

```python
from evals import compress_text, count_tokens

# Compress text with 50% importance cutoff
text = "Your long prompt here..."
compressed, stats = compress_text(text, importance_cutoff=0.5)

print(f"Original: {stats['original_tokens']} tokens")
print(f"Compressed: {stats['compressed_tokens']} tokens")
print(f"Reduction: {stats['reduction_ratio']:.1%}")
```

### Run Evaluation

```bash
# Dry run (no API calls, for testing)
python -m evals.longbench_eval --n 10 --no-api

# Full evaluation with budget guard
python -m evals.longbench_eval --n 30 --cutoffs 0.3 0.9 --budget-usd 10
```

## Project Structure

```
.
├── evals/
│   ├── __init__.py           # Package exports
│   ├── compressor.py         # Compression module with public API
│   └── longbench_eval.py     # Evaluation framework + CLI
├── model.ipynb               # Interactive demo notebook
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── .gitignore
└── runs/                     # Generated at runtime
    ├── cache/                # API response cache
    ├── results.json          # Experiment results
    └── results.csv           # Results in CSV format
```

## Compression Module

### API Reference

#### `compress_text(prompt, importance_cutoff=0.5, **kwargs) -> Tuple[str, dict]`

Compress a raw text prompt.

**Parameters:**
- `prompt` (str): Input text to compress
- `importance_cutoff` (float): Compression aggressiveness, 0.0-1.0. Higher = more aggressive
- `target_tokens` (int, optional): Target token count

**Returns:**
- `compressed` (str): Compressed text
- `stats` (dict): Compression statistics
  - `original_tokens`: Token count before compression
  - `compressed_tokens`: Token count after compression
  - `reduction_ratio`: Fraction of tokens removed
  - `method`: Compression method used

#### `compress_messages(messages, importance_cutoff=0.5, **kwargs) -> Tuple[list, dict]`

Compress chat-style messages while protecting system prompts and the last user message.

**Parameters:**
- `messages` (list): List of message dicts with `role` and `content`
- `importance_cutoff` (float): Compression aggressiveness

**Returns:**
- `compressed_messages` (list): Compressed message list
- `stats` (dict): Aggregated compression statistics

#### `count_tokens(text) -> int`

Count tokens using tiktoken (o200k_base) with fallback heuristic.

#### `identity_compressor(prompt) -> Tuple[str, dict]`

Baseline compressor that returns input unchanged. Used for control condition.

### Compression Strategies

The compressor applies these strategies:

1. **Filler Removal** - Removes phrases like "please note that", "in other words", etc.
2. **Whitespace Normalization** - Consolidates multiple spaces/newlines
3. **Importance-Based Truncation** - Keeps first and last paragraphs, samples from middle
4. **Protected Content** - Preserves code blocks, JSON, quoted strings, URLs, dates

### Example

```python
from evals import compress_text

text = """
Please note that this is a very important document. It is important to note that
we have several key points to discuss. As mentioned earlier, the system requires
authentication using JWT tokens.

```python
def authenticate(token):
    return jwt.decode(token, SECRET_KEY)
```

In conclusion, security is paramount. The API endpoint is https://api.example.com/v1.
"""

compressed, stats = compress_text(text, importance_cutoff=0.5)
print(f"Reduced from {stats['original_tokens']} to {stats['compressed_tokens']} tokens")
# Output: Reduced from 89 to 52 tokens
```

## Evaluation Module

### Experimental Design

The evaluation runs a mini version of The Token Company's LongBench v2 experiment:

**Conditions:**
- `baseline` - No compression (control)
- `cutoff=0.3` - Light compression
- `cutoff=0.9` - Aggressive compression

**Dataset:**
- Source: [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)
- Format: Multiple-choice questions with long context
- Sampling: Stratified by domain and length category

**Metrics:**
- Accuracy (correct answers / valid predictions)
- Invalid rate (unparseable responses)
- Token reduction vs baseline
- Cost per condition

### CLI Usage

```bash
python -m evals.longbench_eval [OPTIONS]
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openai/gpt-4o-mini` | OpenRouter model ID |
| `--n` | `30` | Number of examples to sample |
| `--seed` | `42` | Random seed for reproducibility |
| `--cutoffs` | `0.3 0.9` | Compression cutoff values (space-separated) |
| `--budget-usd` | `10.0` | Maximum budget in USD |
| `--max-context-tokens-for-sampling` | `60000` | Filter examples to this context limit |
| `--max-output-tokens` | `8` | Max tokens for model response |
| `--price-input-per-million` | `0.15` | Input token price (per million) |
| `--price-output-per-million` | `0.60` | Output token price (per million) |
| `--cache-dir` | `runs/cache` | Directory for caching API responses |
| `--results-dir` | `runs` | Directory for saving results |
| `--no-api` | `False` | Dry run mode (no API calls) |

#### Examples

```bash
# Quick test with 5 examples
python -m evals.longbench_eval --n 5 --cutoffs 0.5 --budget-usd 1

# Full run with custom model
python -m evals.longbench_eval --model anthropic/claude-3-haiku --n 50

# Dry run for debugging
python -m evals.longbench_eval --n 10 --no-api

# Custom pricing for different model
python -m evals.longbench_eval \
  --model openai/gpt-4o \
  --price-input-per-million 2.50 \
  --price-output-per-million 10.00 \
  --budget-usd 20
```

### Programmatic API

```python
from evals import run_experiment, compress_text

results = run_experiment(
    compressor_fn=compress_text,
    cutoffs=[0.3, 0.5, 0.9],
    model="openai/gpt-4o-mini",
    n=30,
    seed=42,
    budget_usd=10.0,
    no_api=False,
    openrouter_api_key="your-key-here",  # or set OPENROUTER_API_KEY env var
)

# Access results
for condition in results['conditions']:
    print(f"{condition['condition']}: {condition['accuracy']:.2%} accuracy")
```

### Output Format

#### Console Output

```
====================================================================================================
RESULTS
====================================================================================================
Condition       | Accuracy |  Δ vs Base |          95% CI | Token Red. | Invalid |     Cost
----------------------------------------------------------------------------------------------------
baseline        |    0.733 |        --- |             --- |       0.0% |    0.0% | $ 0.0234
cutoff=0.3      |    0.700 |     -0.033 | [-0.156, +0.089] |      12.4% |    0.0% | $ 0.0205
cutoff=0.9      |    0.633 |     -0.100 | [-0.245, +0.045] |      35.7% |    3.3% | $ 0.0150
====================================================================================================
```

#### JSON Output (`runs/results.json`)

```json
{
  "config": {
    "model": "openai/gpt-4o-mini",
    "n": 30,
    "seed": 42,
    "cutoffs": [0.3, 0.9],
    "budget_usd": 10.0
  },
  "conditions": [
    {
      "condition": "baseline",
      "accuracy": 0.733,
      "invalid_rate": 0.0,
      "mean_input_tokens": 15234.5,
      "token_reduction_vs_baseline": 0.0,
      "total_cost": 0.0234
    }
  ],
  "total_cost": 0.0589
}
```

### Budget Guard

The evaluation automatically adjusts sample size if the estimated cost exceeds the budget:

```
WARNING: Budget exceeded! Reducing sample size from 30 to 18
New estimated cost: $9.87
```

This ensures you never accidentally overspend while running experiments.

### Caching

API responses are cached in `runs/cache/` using a hash of:
- Model name
- Condition name
- Cutoff value
- Example ID
- Temperature
- Prompt hash

Cached results are reused automatically, so you can:
- Resume interrupted runs
- Re-run with different conditions without re-querying cached examples
- Safely experiment without cost concerns for repeated queries

### Statistical Analysis

The evaluation uses bootstrap resampling (10,000 iterations) to compute:

- **95% Confidence Intervals** for accuracy differences vs baseline
- **P(better)** - Probability that the treatment outperforms baseline

This provides robust statistical inference without requiring multiple full runs.

## Demo Notebook

The `model.ipynb` notebook provides an interactive walkthrough:

1. **Setup** - Install dependencies, configure API key
2. **Compression Demo** - Compress a LongBench example, compare cutoff levels
3. **Evaluation** - Run dry-run and real evaluations
4. **Visualization** - Display results table and plots

To run:
```bash
jupyter notebook model.ipynb
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### OpenRouter Setup

1. Get an API key from [OpenRouter](https://openrouter.ai/keys)
2. Add credits to your account
3. Set the API key in `.env` or pass via `openrouter_api_key` parameter

### Supported Models

Any model available on OpenRouter works. Common choices:

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| `openai/gpt-4o-mini` | $0.15 | $0.60 | Default, good balance |
| `openai/gpt-4o` | $2.50 | $10.00 | Higher quality |
| `anthropic/claude-3-haiku` | $0.25 | $1.25 | Fast, cheap |
| `anthropic/claude-3-sonnet` | $3.00 | $15.00 | High quality |
| `google/gemini-flash-1.5` | $0.075 | $0.30 | Very cheap |

Adjust `--price-input-per-million` and `--price-output-per-million` accordingly.

## Development

### Adding a Custom Compressor

Create a function matching this signature:

```python
def my_compressor(prompt: str, importance_cutoff: float, **kwargs) -> Tuple[str, dict]:
    """
    Args:
        prompt: Input text
        importance_cutoff: 0.0-1.0, higher = more aggressive
        **kwargs: Additional options

    Returns:
        (compressed_text, stats_dict)
    """
    # Your compression logic here
    compressed = do_compression(prompt, importance_cutoff)

    return compressed, {
        'original_tokens': count_tokens(prompt),
        'compressed_tokens': count_tokens(compressed),
        'reduction_ratio': 1 - (len(compressed) / len(prompt)),
        'method': 'my_method',
    }
```

Then use it in evaluation:

```python
from evals import run_experiment
from my_module import my_compressor

results = run_experiment(
    compressor_fn=my_compressor,
    cutoffs=[0.3, 0.5, 0.7, 0.9],
    n=50,
)
```

### Running Tests

```bash
# Test compressor module
python -c "from evals import compress_text; print(compress_text('Hello world', 0.5))"

# Test CLI help
python -m evals.longbench_eval --help

# Dry run test
python -m evals.longbench_eval --n 3 --no-api
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

Core dependencies:
- `requests` - HTTP client for OpenRouter API
- `python-dotenv` - Environment variable loading
- `tiktoken` - Token counting
- `numpy` - Statistical computations
- `datasets` - HuggingFace datasets for LongBench v2
- `matplotlib` - Visualization (notebook only)

## License

MIT License

## Acknowledgments

- [LongBench v2](https://github.com/THUDM/LongBench) - Evaluation dataset
- [OpenRouter](https://openrouter.ai) - LLM API provider
- [The Token Company](https://thetoken.company) - Inspiration for evaluation methodology
