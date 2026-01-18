#!/usr/bin/env python3
"""
LongBench v2 Evaluation Module

A comprehensive evaluation framework for prompt compression on LongBench v2.
Supports multiple compression conditions with budget-guarded execution,
deterministic sampling, and detailed statistical analysis.

Usage:
    python -m evals.longbench_eval --n 30 --cutoffs 0.3 0.9 --budget-usd 10

    # Dry run (no API calls)
    python -m evals.longbench_eval --n 10 --dry-run

    # With retries for invalid responses
    python -m evals.longbench_eval --n 30 --retry-invalid 2
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import asdict

import requests

# Local imports
from .prompting import (
    build_prompt_from_example,
    parse_answer,
    build_repair_prompt,
    validate_compressed_prompt,
    ensure_final_instruction,
)
from .stats import (
    bootstrap_paired_diff,
    mcnemar_test,
    compute_accuracy_strict,
    compute_accuracy_valid_only,
    compute_invalid_rate,
)
from .io_utils import (
    ResultCache,
    CacheKey,
    ArtifactWriter,
    ExampleResult,
    ConditionMetrics,
    compute_prompt_hash,
    save_results_json,
    save_results_csv,
    save_examples_csv,
    print_results_table,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try tiktoken
TIKTOKEN_AVAILABLE = False
_encoder = None
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    try:
        _encoder = tiktoken.get_encoding("o200k_base")
    except Exception:
        _encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    logger.warning("tiktoken not available, using heuristic token counting")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE and _encoder:
        try:
            return len(_encoder.encode(text))
        except ValueError:
            pass
    # Fallback heuristic
    words = len(text.split())
    return int(words * 1.3)


# =============================================================================
# OpenRouter Client
# =============================================================================

class OpenRouterClient:
    """Client for OpenRouter API with retry logic."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/longbench-eval",
            "X-Title": "LongBench v2 Evaluation",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 60,
    ) -> Dict:
        """Make chat completion request with retries."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt, delay in enumerate(self.RETRY_DELAYS):
            try:
                resp = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                )

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    logger.warning(f"Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                elif resp.status_code >= 500:
                    logger.warning(f"Server error {resp.status_code}, retrying...")
                    time.sleep(delay)
                else:
                    raise Exception(f"API error {resp.status_code}: {resp.text[:200]}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout, attempt {attempt + 1}/{self.MAX_RETRIES}")
                last_error = "timeout"
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error: {e}")
                last_error = str(e)
                time.sleep(delay)

        raise Exception(f"Failed after {self.MAX_RETRIES} retries: {last_error}")


# =============================================================================
# Dataset Loading
# =============================================================================

def load_longbench_dataset(
    max_context_tokens: int,
    n: int,
    seed: int,
) -> Tuple[List[Dict], int]:
    """
    Load and sample from LongBench v2 with stratification.

    Returns:
        Tuple of (selected_examples, eligible_pool_size)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading LongBench v2 dataset...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)

    # Collect and filter
    domain_length_buckets: Dict[str, List[Dict]] = defaultdict(list)
    total_scanned = 0

    for example in dataset:
        total_scanned += 1
        context = example.get('context', '')
        ctx_tokens = count_tokens(context)

        if ctx_tokens <= max_context_tokens:
            example['_context_tokens'] = ctx_tokens
            domain = example.get('domain', 'unknown')
            length = example.get('length', 'unknown')
            key = f"{domain}|{length}"
            domain_length_buckets[key].append(example)

        # Early exit after scanning enough
        if sum(len(v) for v in domain_length_buckets.values()) >= n * 10:
            break

    eligible_pool_size = sum(len(v) for v in domain_length_buckets.values())
    logger.info(f"Scanned {total_scanned} examples, {eligible_pool_size} eligible across {len(domain_length_buckets)} strata")

    if eligible_pool_size == 0:
        raise ValueError("No examples found within token limit")

    # Stratified round-robin sampling
    random.seed(seed)
    for bucket in domain_length_buckets.values():
        random.shuffle(bucket)

    selected = []
    bucket_iters = {k: iter(v) for k, v in domain_length_buckets.items()}
    keys = list(bucket_iters.keys())

    while len(selected) < n and keys:
        for key in list(keys):
            if len(selected) >= n:
                break
            try:
                selected.append(next(bucket_iters[key]))
            except StopIteration:
                keys.remove(key)

    random.shuffle(selected)
    logger.info(f"Selected {len(selected)} examples")
    return selected, eligible_pool_size


# =============================================================================
# Budget Estimation
# =============================================================================

def estimate_cost(
    examples: List[Dict],
    conditions: List[Tuple[str, float, Optional[Callable]]],
    compressor_fn: Optional[Callable],
    max_output_tokens: int,
    max_retries: int,
    price_in: float,
    price_out: float,
    compressor_mode: str = "ml",
    compressor_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate maximum possible cost including retries.

    Returns:
        Tuple of (total_estimated, per_condition_costs)
    """
    compressor_kwargs = compressor_kwargs or {}
    per_cond = {}
    total = 0.0

    for cond_name, cutoff, _ in conditions:
        cond_cost = 0.0

        for example in examples:
            context = example.get('context', '')
            prompt = build_prompt_from_example(example, context)

            # Apply compression for non-baseline (use same path as runtime)
            if cutoff > 0 and compressor_fn:
                query_text = example.get("question", "") or example.get("input", "") or ""
                compressed, _ = compressor_fn(
                    prompt, 
                    cutoff, 
                    mode=compressor_mode,
                    query_text=query_text,
                    **compressor_kwargs,
                )
                input_tokens = count_tokens(compressed)
            else:
                input_tokens = count_tokens(prompt)

            # Cost per request (with retry upper bound)
            requests_per_example = 1 + max_retries
            input_cost = (input_tokens / 1_000_000) * price_in * requests_per_example
            output_cost = (max_output_tokens / 1_000_000) * price_out * requests_per_example
            cond_cost += input_cost + output_cost

        per_cond[cond_name] = cond_cost
        total += cond_cost

    return total, per_cond


def adjust_sample_size_for_budget(
    examples: List[Dict],
    conditions: List[Tuple[str, float, Optional[Callable]]],
    compressor_fn: Optional[Callable],
    max_output_tokens: int,
    max_retries: int,
    price_in: float,
    price_out: float,
    budget_usd: float,
    compressor_mode: str = "ml",
    compressor_kwargs: Optional[Dict[str, Any]] = None,
) -> int:
    """Find largest N that fits within budget."""
    n = len(examples)
    while n > 1:
        subset = examples[:n]
        estimated, _ = estimate_cost(
            subset, conditions, compressor_fn,
            max_output_tokens, max_retries, price_in, price_out,
            compressor_mode, compressor_kwargs
        )
        if estimated <= budget_usd:
            return n
        n = max(1, n - 5)
    return n


# =============================================================================
# Identity Compressor
# =============================================================================

def identity_compressor(prompt: str, importance_cutoff: float = 0.0, **kwargs) -> Tuple[str, dict]:
    """Identity compressor for baseline - returns input unchanged."""
    tokens = count_tokens(prompt)
    return prompt, {
        'original_tokens': tokens,
        'compressed_tokens': tokens,
        'reduction_ratio': 0.0,
        'method': 'identity',
    }


# =============================================================================
# Main Evaluation
# =============================================================================

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
    price_in: float = 0.15,
    price_out: float = 0.60,
    retry_invalid: int = 1,
    cache_dir: str = "runs/cache",
    results_dir: str = "runs",
    dry_run: bool = False,
    openrouter_api_key: Optional[str] = None,
    write_artifacts: bool = True,
    compressor_mode: str = "ml",
    compressor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run LongBench v2 evaluation experiment.

    Args:
        compressor_fn: Compression function (prompt, cutoff) -> (compressed, stats)
        cutoffs: List of cutoff values for compression conditions
        model: OpenRouter model ID
        n: Number of examples to sample
        seed: Random seed
        budget_usd: Maximum budget in USD
        max_context_tokens_for_sampling: Filter contexts to this limit
        max_output_tokens: Max tokens for model output
        temperature: Model temperature (use 0 for determinism)
        price_in: Price per million input tokens
        price_out: Price per million output tokens
        retry_invalid: Max retries for invalid responses
        cache_dir: Cache directory
        results_dir: Results directory
        dry_run: If True, no API calls
        openrouter_api_key: API key (or from env)
        write_artifacts: If True, write debug artifacts
        compressor_mode: Compression mode (ml, hybrid, heuristic, identity)
        compressor_kwargs: Additional kwargs for compressor

    Returns:
        Results dictionary
    """
    compressor_kwargs = compressor_kwargs or {}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

    # Setup directories
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cache = ResultCache(cache_dir)
    artifact_writer = ArtifactWriter(results_dir, run_id) if write_artifacts else None

    # Define conditions: baseline + compressed
    conditions: List[Tuple[str, float, Optional[Callable]]] = [
        ("baseline", 0.0, None),
    ]
    for cutoff in cutoffs:
        conditions.append((f"cutoff={cutoff}", cutoff, compressor_fn))

    # Load dataset
    examples, eligible_pool_size = load_longbench_dataset(
        max_context_tokens=max_context_tokens_for_sampling,
        n=n,
        seed=seed,
    )

    if not examples:
        return {"error": "No examples loaded"}

    # Estimate cost and adjust N if needed
    estimated_cost, per_cond_cost = estimate_cost(
        examples, conditions, compressor_fn,
        max_output_tokens, retry_invalid, price_in, price_out,
        compressor_mode, compressor_kwargs
    )
    logger.info(f"Estimated max cost for {len(examples)} examples: ${estimated_cost:.4f}")

    if estimated_cost > budget_usd:
        new_n = adjust_sample_size_for_budget(
            examples, conditions, compressor_fn,
            max_output_tokens, retry_invalid, price_in, price_out, budget_usd,
            compressor_mode, compressor_kwargs
        )
        logger.warning(f"Budget exceeded. Reducing N from {len(examples)} to {new_n}")
        examples = examples[:new_n]
        estimated_cost, per_cond_cost = estimate_cost(
            examples, conditions, compressor_fn,
            max_output_tokens, retry_invalid, price_in, price_out,
            compressor_mode, compressor_kwargs
        )
        logger.info(f"New estimated cost: ${estimated_cost:.4f}")

    final_n = len(examples)
    logger.info(f"Final sample size: {final_n}")

    # Setup client
    client = None
    if not dry_run and api_key:
        client = OpenRouterClient(api_key)
    elif not dry_run:
        logger.warning("No API key, switching to dry-run mode")
        dry_run = True

    # Run evaluation
    all_results: List[ExampleResult] = []
    cumulative_cost = 0.0

    for cond_name, cutoff, cond_compressor in conditions:
        logger.info(f"Running condition: {cond_name}")
        is_baseline = (cond_name == "baseline")

        for i, example in enumerate(examples):
            example_id = example.get('_id', str(i))
            context = example.get('context', '')
            gold = example.get('answer', '').upper()

            # Build baseline prompt
            baseline_prompt = build_prompt_from_example(example, context)

            # Extract query_text for relevance scoring
            query_text = (
                example.get("question", "") or 
                example.get("input", "") or 
                example.get("task", "") or
                ""
            )
            # If no explicit query, use the last 500 chars of the prompt as query hint
            if not query_text:
                query_text = baseline_prompt.strip()[-500:]
            
            # Apply compression if not baseline
            if is_baseline:
                final_prompt = baseline_prompt
                fallback_used = False
                comp_stats = {'original_tokens': count_tokens(baseline_prompt), 'compressed_tokens': count_tokens(baseline_prompt)}
            else:
                compressed, comp_stats = compressor_fn(
                    baseline_prompt, 
                    cutoff,
                    mode=compressor_mode,
                    query_text=query_text,
                    use_embeddings=True,
                    cache_dir="runs/emb_cache",
                    **compressor_kwargs,
                )
                # Validate compressed prompt structure
                valid, validation_details = validate_compressed_prompt(compressed, example)
                if not valid:
                    # Fail open to baseline
                    logger.debug(f"Compression validation failed for {example_id}, using baseline")
                    final_prompt = baseline_prompt
                    fallback_used = True
                else:
                    # Ensure final instruction is present
                    final_prompt = ensure_final_instruction(compressed)
                    fallback_used = False

            input_tokens = count_tokens(final_prompt)
            prompt_hash = compute_prompt_hash(final_prompt)

            # Write prompt artifact
            if artifact_writer:
                artifact_writer.write_prompt(example_id, cond_name, final_prompt, is_baseline=is_baseline)
                if not is_baseline:
                    artifact_writer.write_prompt(example_id, cond_name, baseline_prompt, is_baseline=True)

            # Check cache
            cache_key = CacheKey(
                model=model,
                condition=cond_name,
                cutoff=cutoff,
                example_id=example_id,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                prompt_hash=prompt_hash,
                retry_index=0,
            )

            cached = cache.get(cache_key)
            if cached:
                result = ExampleResult(
                    example_id=example_id,
                    condition=cond_name,
                    cutoff=cutoff,
                    predicted=cached.get('predicted'),
                    gold=gold,
                    is_correct=cached.get('is_correct', False),
                    is_valid=cached.get('is_valid', False),
                    input_tokens=cached.get('input_tokens', input_tokens),
                    output_tokens=cached.get('output_tokens', 0),
                    total_tokens=cached.get('input_tokens', input_tokens) + cached.get('output_tokens', 0),
                    cost_usd=cached.get('cost_usd', 0.0),
                    cached=True,
                    retry_count=cached.get('retry_count', 0),
                    invalid_before_retry=cached.get('invalid_before_retry', False),
                    invalid_after_retry=cached.get('invalid_after_retry', not cached.get('is_valid', False)),
                    fallback_used=cached.get('fallback_used', fallback_used),
                    raw_response=cached.get('raw_response', ''),
                )
                all_results.append(result)
                cumulative_cost += result.cost_usd
                continue

            if dry_run:
                # Simulate
                result = ExampleResult(
                    example_id=example_id,
                    condition=cond_name,
                    cutoff=cutoff,
                    predicted=None,
                    gold=gold,
                    is_correct=False,
                    is_valid=False,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    total_tokens=input_tokens,
                    cost_usd=0.0,
                    cached=False,
                    retry_count=0,
                    invalid_before_retry=True,
                    invalid_after_retry=True,
                    fallback_used=fallback_used,
                    raw_response="",
                )
                all_results.append(result)
                continue

            # Make API call(s)
            messages = [{"role": "user", "content": final_prompt}]
            raw_response = ""
            predicted = None
            is_valid = False
            output_tokens = 0
            retry_count = 0
            invalid_before_retry = False

            for attempt in range(1 + retry_invalid):
                try:
                    response = client.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                    )

                    raw_response = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    usage = response.get('usage', {})
                    output_tokens = usage.get('completion_tokens', max_output_tokens)

                    predicted = parse_answer(raw_response)
                    is_valid = predicted is not None

                    if artifact_writer:
                        artifact_writer.write_response(example_id, cond_name, raw_response, retry_index=attempt)

                    if is_valid:
                        break

                    # Mark first attempt invalid
                    if attempt == 0:
                        invalid_before_retry = True

                    # Retry with repair prompt
                    if attempt < retry_invalid:
                        retry_count += 1
                        repair = build_repair_prompt(raw_response)
                        messages.append({"role": "assistant", "content": raw_response})
                        messages.append({"role": "user", "content": repair})

                except Exception as e:
                    logger.error(f"API error for {example_id}: {e}")
                    break

            is_correct = is_valid and predicted == gold
            cost = (
                (input_tokens / 1_000_000) * price_in +
                (output_tokens / 1_000_000) * price_out
            )

            result = ExampleResult(
                example_id=example_id,
                condition=cond_name,
                cutoff=cutoff,
                predicted=predicted,
                gold=gold,
                is_correct=is_correct,
                is_valid=is_valid,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost,
                cached=False,
                retry_count=retry_count,
                invalid_before_retry=invalid_before_retry,
                invalid_after_retry=not is_valid,
                fallback_used=fallback_used,
                raw_response=raw_response,
            )

            # Cache result
            cache.set(cache_key, {
                'predicted': predicted,
                'is_correct': is_correct,
                'is_valid': is_valid,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': cost,
                'retry_count': retry_count,
                'invalid_before_retry': invalid_before_retry,
                'fallback_used': fallback_used,
                'raw_response': raw_response,
            })

            # Write parsed answer artifact
            if artifact_writer:
                artifact_writer.write_parsed_answer(example_id, cond_name, {
                    'valid': is_valid,
                    'predicted': predicted,
                    'gold': gold,
                    'correct': is_correct,
                    'retry_count': retry_count,
                    'fallback_used': fallback_used,
                })

            all_results.append(result)
            cumulative_cost += cost

            if (i + 1) % 5 == 0:
                logger.info(f"  [{cond_name}] {i+1}/{final_n} | Cost so far: ${cumulative_cost:.4f}")

    # Aggregate metrics
    condition_metrics = aggregate_metrics(all_results, conditions, seed)

    # Print table
    print_results_table(condition_metrics)

    # Build config dict
    config = {
        'model': model,
        'n': final_n,
        'seed': seed,
        'cutoffs': cutoffs,
        'budget_usd': budget_usd,
        'max_context_tokens_for_sampling': max_context_tokens_for_sampling,
        'max_output_tokens': max_output_tokens,
        'temperature': temperature,
        'price_in': price_in,
        'price_out': price_out,
        'retry_invalid': retry_invalid,
        'dry_run': dry_run,
        'compressor_mode': compressor_mode,
    }

    run_metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'eligible_pool_size': eligible_pool_size,
        'final_sample_size': final_n,
        'total_cost_usd': cumulative_cost,
        'cache_stats': cache.get_stats(),
    }

    # Save results
    save_results_json(results_dir, config, condition_metrics, all_results, run_metadata)
    save_results_csv(results_dir, condition_metrics)
    save_examples_csv(results_dir, all_results)

    logger.info(f"Results saved to {results_dir}/")
    logger.info(f"Total cost: ${cumulative_cost:.4f}")

    return {
        'config': config,
        'run_metadata': run_metadata,
        'conditions': [asdict(m) for m in condition_metrics],
        'total_cost': cumulative_cost,
    }


def aggregate_metrics(
    results: List[ExampleResult],
    conditions: List[Tuple[str, float, Optional[Callable]]],
    seed: int,
) -> List[ConditionMetrics]:
    """Aggregate per-example results into condition-level metrics."""
    by_cond: Dict[str, List[ExampleResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    # Get baseline data for comparison
    baseline_results = by_cond.get('baseline', [])
    baseline_correct = [r.is_correct for r in baseline_results]
    baseline_valid = [r.is_valid for r in baseline_results]
    baseline_input_tokens = [r.input_tokens for r in baseline_results]
    baseline_mean_tokens = sum(baseline_input_tokens) / len(baseline_input_tokens) if baseline_input_tokens else 1

    # Build example_id -> baseline result mapping for pairing
    baseline_by_id = {r.example_id: r for r in baseline_results}

    metrics_list = []

    for cond_name, cutoff, _ in conditions:
        cond_results = by_cond.get(cond_name, [])
        if not cond_results:
            continue

        n_examples = len(cond_results)
        correct = [r.is_correct for r in cond_results]
        valid = [r.is_valid for r in cond_results]

        acc_strict = compute_accuracy_strict(correct, valid)
        acc_valid_only = compute_accuracy_valid_only(correct, valid)
        invalid_rate = compute_invalid_rate(valid)

        invalid_before = sum(1 for r in cond_results if r.invalid_before_retry)
        invalid_after = sum(1 for r in cond_results if r.invalid_after_retry)
        fallbacks = sum(1 for r in cond_results if r.fallback_used)

        mean_in = sum(r.input_tokens for r in cond_results) / n_examples
        mean_out = sum(r.output_tokens for r in cond_results) / n_examples
        mean_total = sum(r.total_tokens for r in cond_results) / n_examples
        token_reduction = 1.0 - (mean_in / baseline_mean_tokens) if baseline_mean_tokens > 0 else 0.0

        actual_cost = sum(r.cost_usd for r in cond_results)
        estimated_cost = actual_cost  # Already actual for completed runs

        # Statistical comparison to baseline
        if cond_name != "baseline" and baseline_results:
            # Pair by example_id
            paired_baseline_correct = []
            paired_treatment_correct = []
            for r in cond_results:
                if r.example_id in baseline_by_id:
                    br = baseline_by_id[r.example_id]
                    # Strict: invalid = wrong
                    paired_baseline_correct.append(br.is_correct and br.is_valid)
                    paired_treatment_correct.append(r.is_correct and r.is_valid)

            # Bootstrap paired difference
            baseline_strict = [1.0 if c else 0.0 for c in paired_baseline_correct]
            treatment_strict = [1.0 if c else 0.0 for c in paired_treatment_correct]
            delta, ci_lo, ci_hi, p_better = bootstrap_paired_diff(
                baseline_strict, treatment_strict, seed=seed
            )

            # McNemar test
            chi2, pval, _, _ = mcnemar_test(paired_baseline_correct, paired_treatment_correct)
        else:
            delta, ci_lo, ci_hi, p_better = 0.0, 0.0, 0.0, 0.5
            chi2, pval = 0.0, 1.0

        metrics_list.append(ConditionMetrics(
            condition=cond_name,
            cutoff=cutoff,
            n_examples=n_examples,
            accuracy_strict=acc_strict,
            accuracy_valid_only=acc_valid_only,
            invalid_rate=invalid_rate,
            invalid_before_retry_rate=invalid_before / n_examples if n_examples else 0,
            invalid_after_retry_rate=invalid_after / n_examples if n_examples else 0,
            fallback_rate=fallbacks / n_examples if n_examples else 0,
            mean_input_tokens=mean_in,
            mean_output_tokens=mean_out,
            mean_total_tokens=mean_total,
            token_reduction_vs_baseline=token_reduction,
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=actual_cost,
            delta_strict_vs_baseline=delta,
            delta_ci_lower=ci_lo,
            delta_ci_upper=ci_hi,
            p_better=p_better,
            mcnemar_chi2=chi2,
            mcnemar_pvalue=pval,
        ))

    return metrics_list


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LongBench v2 Evaluation for prompt compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini',
                        help='OpenRouter model ID')
    parser.add_argument('--n', type=int, default=30,
                        help='Number of examples to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[0.3, 0.9],
                        help='Compression cutoff values')
    parser.add_argument('--budget-usd', type=float, default=10.0,
                        help='Maximum budget in USD')
    parser.add_argument('--max-context-tokens-for-sampling', type=int, default=60000,
                        help='Max context tokens for filtering')
    parser.add_argument('--max-output-tokens', type=int, default=8,
                        help='Max output tokens')
    parser.add_argument('--price-in', type=float, default=0.15,
                        help='Price per million input tokens')
    parser.add_argument('--price-out', type=float, default=0.60,
                        help='Price per million output tokens')
    parser.add_argument('--retry-invalid', type=int, default=1,
                        help='Max retries for invalid responses')
    parser.add_argument('--cache-dir', type=str, default='runs/cache',
                        help='Cache directory')
    parser.add_argument('--results-dir', type=str, default='runs',
                        help='Results directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without API calls')
    parser.add_argument('--no-artifacts', action='store_true',
                        help='Skip writing debug artifacts')
    parser.add_argument('--compressor-mode', type=str, default='auto',
                        choices=['auto', 'ml', 'hybrid', 'identity'],
                        help='Compression mode: auto (ML-first with hybrid fallback), ml (LLMLingua-2), hybrid (IR+embeddings), identity (no compression)')

    args = parser.parse_args()

    # Import compressor
    try:
        from .compressor import compress_text
    except ImportError:
        try:
            from compressor import compress_text
        except ImportError:
            logger.error("Could not import compressor. Using identity.")
            compress_text = identity_compressor

    # Run
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
            price_in=args.price_in,
            price_out=args.price_out,
            retry_invalid=args.retry_invalid,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
            dry_run=args.dry_run,
            write_artifacts=not args.no_artifacts,
            compressor_mode=args.compressor_mode,
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
