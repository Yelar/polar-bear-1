#!/usr/bin/env python3
"""
LongBench v2 Evaluation Module

A mini version of The Token Company LongBench v2 experiment for OpenRouter.
Supports multiple compression conditions with budget-guarded execution.

Usage:
    python longbench_eval.py --model openai/gpt-4o-mini --n 30 --seed 42 --cutoffs 0.3 0.9 --budget-usd 10

Author: LLM Input Compressor Team
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Set OPENROUTER_API_KEY in environment.")

# Try to import tiktoken for accurate token counting
TIKTOKEN_AVAILABLE = False
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning("tiktoken not available, using heuristic token counting")

# Try to import numpy for bootstrap
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not available, using stdlib for statistics")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    model: str = "openai/gpt-4o-mini"
    n: int = 30
    seed: int = 42
    cutoffs: List[float] = field(default_factory=lambda: [0.3, 0.9])
    budget_usd: float = 10.0
    max_context_tokens_for_sampling: int = 60_000
    max_output_tokens: int = 8
    temperature: float = 0.0
    price_input_per_million: float = 0.15
    price_output_per_million: float = 0.60
    cache_dir: str = "runs/cache"
    results_dir: str = "runs"
    no_api: bool = False
    openrouter_api_key: Optional[str] = None

    def __post_init__(self):
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# =============================================================================
# Token Counting
# =============================================================================

class TokenCounter:
    """Token counter with tiktoken or fallback."""

    def __init__(self, encoding_name: str = "o200k_base"):
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except ValueError as e:
                if "disallowed special token" in str(e):
                    # Handle special token error by using fallback
                    words = text.split()
                    punct = len(re.findall(r'[.,!?;:"\'()\[\]{}]', text))
                    return int((len(words) + punct) * 1.3)
                else:
                    raise
        # Fallback heuristic
        words = text.split()
        punct = len(re.findall(r'[.,!?;:"\'()\[\]{}]', text))
        return int((len(words) + punct) * 1.3)


_token_counter = TokenCounter()


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return _token_counter.count(text)


# =============================================================================
# Caching
# =============================================================================

class ResultCache:
    """Filesystem cache for API results."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(
        self,
        model: str,
        condition: str,
        cutoff: float,
        example_id: str,
        temperature: float,
        prompt_hash: str
    ) -> str:
        """Create cache key."""
        key_str = f"{model}|{condition}|{cutoff}|{example_id}|{temperature}|{prompt_hash}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(
        self,
        model: str,
        condition: str,
        cutoff: float,
        example_id: str,
        temperature: float,
        prompt_hash: str
    ) -> Optional[Dict]:
        """Get cached result if exists."""
        key = self._make_key(model, condition, cutoff, example_id, temperature, prompt_hash)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(
        self,
        model: str,
        condition: str,
        cutoff: float,
        example_id: str,
        temperature: float,
        prompt_hash: str,
        result: Dict
    ):
        """Cache a result."""
        key = self._make_key(model, condition, cutoff, example_id, temperature, prompt_hash)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f)


# =============================================================================
# OpenRouter API Client
# =============================================================================

class OpenRouterClient:
    """Client for OpenRouter Chat Completions API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/longbench-eval",
            "X-Title": "LongBench v2 Evaluation"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 60
    ) -> Dict:
        """Make a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited
                    logger.warning(f"Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue
                elif response.status_code >= 500:
                    # Server error
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(delay)
                    continue
                else:
                    # Client error
                    error_msg = response.text[:200]
                    raise Exception(f"API error {response.status_code}: {error_msg}")

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{self.MAX_RETRIES}")
                last_error = "timeout"
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error: {e}")
                last_error = str(e)
                time.sleep(delay)

        raise Exception(f"Failed after {self.MAX_RETRIES} retries: {last_error}")


# =============================================================================
# Answer Parsing
# =============================================================================

def parse_answer(text: str) -> Optional[str]:
    """
    Parse model output to extract A/B/C/D answer.

    Handles formats:
    - "A", "B", "C", "D"
    - "Answer: A", "The answer is B"
    - "(A)", "[A]"
    - "A." or "A:"

    Returns None if no valid answer found.
    """
    if not text:
        return None

    text = text.strip().upper()

    # Pattern: standalone letter A-D
    patterns = [
        r'^([ABCD])$',  # Just the letter
        r'^([ABCD])[.:\)]',  # Letter followed by punctuation
        r'^\(([ABCD])\)',  # (A)
        r'^\[([ABCD])\]',  # [A]
        r'^ANSWER[:\s]+([ABCD])',  # Answer: A
        r'^THE ANSWER IS[:\s]+([ABCD])',  # The answer is A
        r'^([ABCD])\s*$',  # Letter with trailing space
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1)

    # Fallback: find first standalone A-D in text
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)

    return None


# =============================================================================
# Dataset Loading
# =============================================================================

def load_longbench_dataset(
    max_context_tokens: int,
    n: int,
    seed: int,
    streaming: bool = True
) -> List[Dict]:
    """
    Load and sample from LongBench v2 dataset.

    Filters to rows with context <= max_context_tokens.
    Stratifies by domain and length using round-robin.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info("Loading LongBench v2 dataset...")

    # Load dataset
    if streaming:
        dataset = load_dataset("zai-org/LongBench-v2", split="train", streaming=True)
    else:
        dataset = load_dataset("zai-org/LongBench-v2", split="train")

    # Filter and collect examples
    filtered = []
    domain_length_buckets = defaultdict(list)

    for example in dataset:
        context = example.get('context', '')
        ctx_tokens = count_tokens(context)

        if ctx_tokens <= max_context_tokens:
            example['_context_tokens'] = ctx_tokens
            domain = example.get('domain', 'unknown')
            length = example.get('length', 'unknown')
            key = f"{domain}|{length}"
            domain_length_buckets[key].append(example)

        # Early exit if we have enough
        if sum(len(v) for v in domain_length_buckets.values()) >= n * 10:
            break

    if not domain_length_buckets:
        raise ValueError("No examples found within token limit")

    logger.info(f"Found {sum(len(v) for v in domain_length_buckets.values())} eligible examples across {len(domain_length_buckets)} strata")

    # Stratified sampling via round-robin
    random.seed(seed)

    # Shuffle within each bucket
    for key in domain_length_buckets:
        random.shuffle(domain_length_buckets[key])

    # Round-robin selection
    selected = []
    bucket_iters = {k: iter(v) for k, v in domain_length_buckets.items()}
    keys = list(bucket_iters.keys())

    while len(selected) < n and bucket_iters:
        for key in list(keys):
            if len(selected) >= n:
                break
            try:
                selected.append(next(bucket_iters[key]))
            except StopIteration:
                keys.remove(key)

    # Shuffle final selection for randomness
    random.shuffle(selected)

    logger.info(f"Selected {len(selected)} examples")
    return selected


# =============================================================================
# Prompt Building
# =============================================================================

def build_prompt(example: Dict, context: str) -> str:
    """Build the multiple-choice prompt."""
    question = example.get('question', '')
    choice_a = example.get('choice_A', '')
    choice_b = example.get('choice_B', '')
    choice_c = example.get('choice_C', '')
    choice_d = example.get('choice_D', '')

    prompt = f"""Read the following context and answer the multiple-choice question.

Context:
{context}

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer with ONLY the letter (A, B, C, or D) of the correct answer."""

    return prompt


# =============================================================================
# Cost Estimation
# =============================================================================

def estimate_cost(
    examples: List[Dict],
    conditions: List[Tuple[str, float, Callable]],
    config: EvalConfig
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate total cost for all conditions.

    Returns (total_cost, per_condition_costs)
    """
    per_condition = {}
    total = 0.0

    for cond_name, cutoff, compressor_fn in conditions:
        cond_cost = 0.0

        for example in examples:
            context = example.get('context', '')
            prompt = build_prompt(example, context)

            # Apply compression
            if compressor_fn:
                compressed, _ = compressor_fn(prompt, cutoff)
                input_tokens = count_tokens(compressed)
            else:
                input_tokens = count_tokens(prompt)

            # Cost calculation
            input_cost = (input_tokens / 1_000_000) * config.price_input_per_million
            output_cost = (config.max_output_tokens / 1_000_000) * config.price_output_per_million
            cond_cost += input_cost + output_cost

        per_condition[cond_name] = cond_cost
        total += cond_cost

    return total, per_condition


def adjust_sample_size(
    examples: List[Dict],
    conditions: List[Tuple[str, float, Callable]],
    config: EvalConfig
) -> int:
    """Adjust sample size to fit within budget."""
    n = len(examples)

    while n > 1:
        subset = examples[:n]
        estimated, _ = estimate_cost(subset, conditions, config)

        if estimated <= config.budget_usd:
            return n

        n = max(1, n - 5)

    return n


# =============================================================================
# Bootstrap Statistics
# =============================================================================

def bootstrap_ci(
    data: List[float],
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper)
    """
    if not data:
        return 0.0, 0.0, 0.0

    if NUMPY_AVAILABLE:
        np.random.seed(seed)
        data_arr = np.array(data)
        means = []
        for _ in range(n_iterations):
            sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
            means.append(np.mean(sample))
        means = np.array(means)
        alpha = (1 - ci) / 2
        return float(np.mean(data)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))
    else:
        random.seed(seed)
        means = []
        for _ in range(n_iterations):
            sample = random.choices(data, k=len(data))
            means.append(sum(sample) / len(sample))
        means.sort()
        alpha = (1 - ci) / 2
        lower_idx = int(alpha * len(means))
        upper_idx = int((1 - alpha) * len(means))
        return sum(data) / len(data), means[lower_idx], means[upper_idx]


def bootstrap_diff(
    baseline: List[float],
    treatment: List[float],
    n_iterations: int = 10_000,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Bootstrap difference between treatment and baseline.

    Returns (mean_diff, ci_lower, ci_upper, p_better)
    """
    if not baseline or not treatment:
        return 0.0, 0.0, 0.0, 0.0

    if NUMPY_AVAILABLE:
        np.random.seed(seed)
        base_arr = np.array(baseline)
        treat_arr = np.array(treatment)
        diffs = []
        for _ in range(n_iterations):
            base_sample = np.random.choice(base_arr, size=len(base_arr), replace=True)
            treat_sample = np.random.choice(treat_arr, size=len(treat_arr), replace=True)
            diffs.append(np.mean(treat_sample) - np.mean(base_sample))
        diffs = np.array(diffs)
        p_better = float(np.mean(diffs > 0))
        return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), p_better
    else:
        random.seed(seed)
        diffs = []
        for _ in range(n_iterations):
            base_sample = random.choices(baseline, k=len(baseline))
            treat_sample = random.choices(treatment, k=len(treatment))
            diff = sum(treat_sample) / len(treat_sample) - sum(base_sample) / len(base_sample)
            diffs.append(diff)
        diffs.sort()
        p_better = sum(1 for d in diffs if d > 0) / len(diffs)
        return sum(diffs) / len(diffs), diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))], p_better


# =============================================================================
# Main Evaluation
# =============================================================================

@dataclass
class ExampleResult:
    """Result for a single example."""
    example_id: str
    condition: str
    cutoff: float
    predicted: Optional[str]
    correct: str
    is_correct: bool
    is_valid: bool
    input_tokens: int
    output_tokens: int
    cost: float
    cached: bool = False


@dataclass
class ConditionResult:
    """Aggregated results for a condition."""
    condition: str
    cutoff: float
    accuracy: float
    invalid_rate: float
    mean_input_tokens: float
    token_reduction_vs_baseline: float
    total_cost: float
    n_examples: int
    delta_vs_baseline: float = 0.0
    delta_ci_lower: float = 0.0
    delta_ci_upper: float = 0.0
    p_better: float = 0.0


def run_experiment(
    compressor_fn: Callable[[str, float], Tuple[str, dict]],
    cutoffs: List[float],
    model: str = "openai/gpt-4o-mini",
    n: int = 30,
    seed: int = 42,
    budget_usd: float = 10.0,
    max_context_tokens_for_sampling: int = 60_000,
    max_output_tokens: int = 8,
    temperature: float = 0.0,
    price_input_per_million: float = 0.15,
    price_output_per_million: float = 0.60,
    cache_dir: str = "runs/cache",
    results_dir: str = "runs",
    no_api: bool = False,
    openrouter_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the LongBench v2 evaluation experiment.

    Args:
        compressor_fn: Function (prompt, cutoff) -> (compressed, stats)
        cutoffs: List of cutoff values for compression conditions
        model: OpenRouter model ID
        n: Number of examples to sample
        seed: Random seed
        budget_usd: Maximum budget in USD
        max_context_tokens_for_sampling: Filter contexts to this token limit
        max_output_tokens: Max tokens for model output
        temperature: Model temperature
        price_input_per_million: Input token price per million
        price_output_per_million: Output token price per million
        cache_dir: Directory for caching results
        results_dir: Directory for saving results
        no_api: If True, run in dry-run mode without API calls

    Returns:
        Dictionary with results and statistics
    """
    config = EvalConfig(
        model=model,
        n=n,
        seed=seed,
        cutoffs=cutoffs,
        budget_usd=budget_usd,
        max_context_tokens_for_sampling=max_context_tokens_for_sampling,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        price_input_per_million=price_input_per_million,
        price_output_per_million=price_output_per_million,
        cache_dir=cache_dir,
        results_dir=results_dir,
        no_api=no_api,
        openrouter_api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
    )

    # Setup cache
    cache = ResultCache(config.cache_dir)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    # Define conditions
    from compressor import identity_compressor
    conditions = [("baseline", 0.0, identity_compressor)]
    for cutoff in cutoffs:
        conditions.append((f"cutoff={cutoff}", cutoff, compressor_fn))

    # Load dataset
    examples = load_longbench_dataset(
        max_context_tokens=config.max_context_tokens_for_sampling,
        n=config.n,
        seed=config.seed
    )

    if not examples:
        logger.error("No examples loaded!")
        return {"error": "No examples loaded"}

    # Estimate cost and adjust sample size
    estimated_cost, per_cond_cost = estimate_cost(examples, conditions, config)
    logger.info(f"Estimated cost for {len(examples)} examples: ${estimated_cost:.4f}")

    if estimated_cost > config.budget_usd:
        new_n = adjust_sample_size(examples, conditions, config)
        logger.warning(f"Budget exceeded! Reducing sample size from {len(examples)} to {new_n}")
        examples = examples[:new_n]
        estimated_cost, per_cond_cost = estimate_cost(examples, conditions, config)
        logger.info(f"New estimated cost: ${estimated_cost:.4f}")

    if not examples:
        logger.error("Cannot run experiment with 0 examples")
        return {"error": "Budget too low for any samples"}

    # Setup API client
    client = None
    if not config.no_api and config.openrouter_api_key:
        client = OpenRouterClient(config.openrouter_api_key)
    elif not config.no_api:
        logger.warning("No API key found, running in dry-run mode")
        config.no_api = True

    # Run evaluation
    all_results: List[ExampleResult] = []
    cumulative_cost = 0.0

    for cond_name, cutoff, cond_compressor in conditions:
        logger.info(f"Running condition: {cond_name}")

        for i, example in enumerate(examples):
            example_id = example.get('_id', str(i))
            context = example.get('context', '')
            correct_answer = example.get('answer', '').upper()

            # Build and compress prompt
            full_prompt = build_prompt(example, context)

            if cond_compressor:
                compressed_prompt, comp_stats = cond_compressor(full_prompt, cutoff)
            else:
                compressed_prompt = full_prompt
                comp_stats = {'original_tokens': count_tokens(full_prompt), 'compressed_tokens': count_tokens(full_prompt)}

            input_tokens = count_tokens(compressed_prompt)
            prompt_hash = hashlib.sha256(compressed_prompt.encode()).hexdigest()[:16]

            # Check cache
            cached_result = cache.get(
                model=config.model,
                condition=cond_name,
                cutoff=cutoff,
                example_id=example_id,
                temperature=config.temperature,
                prompt_hash=prompt_hash
            )

            if cached_result:
                result = ExampleResult(
                    example_id=example_id,
                    condition=cond_name,
                    cutoff=cutoff,
                    predicted=cached_result.get('predicted'),
                    correct=correct_answer,
                    is_correct=cached_result.get('is_correct', False),
                    is_valid=cached_result.get('is_valid', False),
                    input_tokens=cached_result.get('input_tokens', input_tokens),
                    output_tokens=cached_result.get('output_tokens', config.max_output_tokens),
                    cost=cached_result.get('cost', 0.0),
                    cached=True
                )
            elif config.no_api:
                # Dry run - simulate response
                result = ExampleResult(
                    example_id=example_id,
                    condition=cond_name,
                    cutoff=cutoff,
                    predicted=None,
                    correct=correct_answer,
                    is_correct=False,
                    is_valid=False,
                    input_tokens=input_tokens,
                    output_tokens=config.max_output_tokens,
                    cost=0.0,
                    cached=False
                )
            else:
                # Make API call
                messages = [{"role": "user", "content": compressed_prompt}]

                try:
                    response = client.chat_completion(
                        messages=messages,
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_output_tokens
                    )

                    # Extract response
                    output_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    usage = response.get('usage', {})
                    output_tokens = usage.get('completion_tokens', config.max_output_tokens)

                    # Parse answer
                    predicted = parse_answer(output_text)
                    is_valid = predicted is not None
                    is_correct = is_valid and predicted == correct_answer

                    # Calculate cost
                    cost = (
                        (input_tokens / 1_000_000) * config.price_input_per_million +
                        (output_tokens / 1_000_000) * config.price_output_per_million
                    )

                    result = ExampleResult(
                        example_id=example_id,
                        condition=cond_name,
                        cutoff=cutoff,
                        predicted=predicted,
                        correct=correct_answer,
                        is_correct=is_correct,
                        is_valid=is_valid,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        cached=False
                    )

                    # Cache the result
                    cache.set(
                        model=config.model,
                        condition=cond_name,
                        cutoff=cutoff,
                        example_id=example_id,
                        temperature=config.temperature,
                        prompt_hash=prompt_hash,
                        result={
                            'predicted': predicted,
                            'is_correct': is_correct,
                            'is_valid': is_valid,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'cost': cost,
                        }
                    )

                except Exception as e:
                    logger.error(f"API error for example {example_id}: {e}")
                    result = ExampleResult(
                        example_id=example_id,
                        condition=cond_name,
                        cutoff=cutoff,
                        predicted=None,
                        correct=correct_answer,
                        is_correct=False,
                        is_valid=False,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        cost=0.0,
                        cached=False
                    )

            all_results.append(result)
            cumulative_cost += result.cost

            # Progress logging
            if (i + 1) % 5 == 0:
                logger.info(f"  [{cond_name}] {i+1}/{len(examples)} | Cumulative cost: ${cumulative_cost:.4f}")

    # Aggregate results
    condition_results = aggregate_results(all_results, config)

    # Print results table
    print_results_table(condition_results)

    # Save results
    results_data = {
        'config': asdict(config),
        'conditions': [asdict(r) for r in condition_results],
        'examples': [asdict(r) for r in all_results],
        'total_cost': cumulative_cost,
    }

    results_json = Path(config.results_dir) / "results.json"
    with open(results_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to {results_json}")

    # Save CSV
    results_csv = Path(config.results_dir) / "results.csv"
    save_results_csv(condition_results, results_csv)
    logger.info(f"CSV saved to {results_csv}")

    return results_data


def aggregate_results(
    results: List[ExampleResult],
    config: EvalConfig
) -> List[ConditionResult]:
    """Aggregate individual results into condition-level statistics."""
    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    # Get baseline results for comparison
    baseline_results = by_condition.get('baseline', [])
    baseline_correct = [1.0 if r.is_correct else 0.0 for r in baseline_results if r.is_valid]
    baseline_tokens = [r.input_tokens for r in baseline_results]
    baseline_mean_tokens = sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else 0

    condition_results = []

    for cond_name, cond_results in by_condition.items():
        valid_results = [r for r in cond_results if r.is_valid]
        correct_list = [1.0 if r.is_correct else 0.0 for r in cond_results if r.is_valid]

        accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
        invalid_rate = 1.0 - (len(valid_results) / len(cond_results)) if cond_results else 0.0
        mean_tokens = sum(r.input_tokens for r in cond_results) / len(cond_results) if cond_results else 0
        token_reduction = 1.0 - (mean_tokens / baseline_mean_tokens) if baseline_mean_tokens > 0 else 0.0
        total_cost = sum(r.cost for r in cond_results)
        cutoff = cond_results[0].cutoff if cond_results else 0.0

        # Bootstrap comparison to baseline
        if cond_name != 'baseline' and baseline_correct and correct_list:
            delta, ci_lower, ci_upper, p_better = bootstrap_diff(
                baseline_correct, correct_list, seed=config.seed
            )
        else:
            delta, ci_lower, ci_upper, p_better = 0.0, 0.0, 0.0, 0.5

        condition_results.append(ConditionResult(
            condition=cond_name,
            cutoff=cutoff,
            accuracy=accuracy,
            invalid_rate=invalid_rate,
            mean_input_tokens=mean_tokens,
            token_reduction_vs_baseline=token_reduction,
            total_cost=total_cost,
            n_examples=len(cond_results),
            delta_vs_baseline=delta,
            delta_ci_lower=ci_lower,
            delta_ci_upper=ci_upper,
            p_better=p_better,
        ))

    return condition_results


def print_results_table(results: List[ConditionResult]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"{'Condition':<15} | {'Accuracy':>8} | {'Δ vs Base':>10} | {'95% CI':>15} | {'Token Red.':>10} | {'Invalid':>8} | {'Cost':>8}")
    print("-" * 100)

    for r in results:
        ci_str = f"[{r.delta_ci_lower:+.3f}, {r.delta_ci_upper:+.3f}]" if r.condition != 'baseline' else "---"
        delta_str = f"{r.delta_vs_baseline:+.3f}" if r.condition != 'baseline' else "---"
        print(f"{r.condition:<15} | {r.accuracy:>8.3f} | {delta_str:>10} | {ci_str:>15} | {r.token_reduction_vs_baseline:>9.1%} | {r.invalid_rate:>7.1%} | ${r.total_cost:>7.4f}")

    print("=" * 100)


def save_results_csv(results: List[ConditionResult], path: Path):
    """Save results as CSV."""
    headers = ['condition', 'cutoff', 'accuracy', 'delta_vs_baseline', 'ci_lower', 'ci_upper',
               'p_better', 'token_reduction', 'invalid_rate', 'cost', 'n_examples']

    with open(path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for r in results:
            row = [
                r.condition, str(r.cutoff), f"{r.accuracy:.4f}", f"{r.delta_vs_baseline:.4f}",
                f"{r.delta_ci_lower:.4f}", f"{r.delta_ci_upper:.4f}", f"{r.p_better:.4f}",
                f"{r.token_reduction_vs_baseline:.4f}", f"{r.invalid_rate:.4f}",
                f"{r.total_cost:.6f}", str(r.n_examples)
            ]
            f.write(','.join(row) + '\n')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LongBench v2 Evaluation for prompt compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini',
                        help='OpenRouter model ID')
    parser.add_argument('--n', type=int, default=30,
                        help='Number of examples to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[0.3, 0.9],
                        help='Compression cutoff values')
    parser.add_argument('--budget-usd', type=float, default=10.0,
                        help='Maximum budget in USD')
    parser.add_argument('--max-context-tokens-for-sampling', type=int, default=60000,
                        help='Maximum context tokens for filtering examples')
    parser.add_argument('--max-output-tokens', type=int, default=8,
                        help='Maximum output tokens from model')
    parser.add_argument('--price-input-per-million', type=float, default=0.15,
                        help='Price per million input tokens')
    parser.add_argument('--price-output-per-million', type=float, default=0.60,
                        help='Price per million output tokens')
    parser.add_argument('--cache-dir', type=str, default='runs/cache',
                        help='Directory for caching API results')
    parser.add_argument('--results-dir', type=str, default='runs',
                        help='Directory for saving results')
    parser.add_argument('--no-api', action='store_true',
                        help='Run in dry-run mode without API calls')

    args = parser.parse_args()

    # Import compressor
    try:
        from compressor import compress_text
    except ImportError:
        logger.error("Could not import compressor.py. Make sure it exists in the same directory.")
        sys.exit(1)

    # Run experiment
    try:
        results = run_experiment(
            compressor_fn=compress_text,
            cutoffs=args.cutoffs,
            model=args.model,
            n=args.n,
            seed=args.seed,
            budget_usd=args.budget_usd,
            max_context_tokens_for_sampling=args.max_context_tokens_for_sampling,
            max_output_tokens=args.max_output_tokens,
            price_input_per_million=args.price_input_per_million,
            price_output_per_million=args.price_output_per_million,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
            no_api=args.no_api,
        )

        if 'error' in results:
            logger.error(results['error'])
            sys.exit(1)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
