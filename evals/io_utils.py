"""
I/O utilities and lightweight data structures for evaluation runs.
"""

import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class CacheKey:
    model: str
    condition: str
    cutoff: float
    example_id: str
    temperature: float
    max_output_tokens: int
    prompt_hash: str
    retry_index: int


@dataclass
class ExampleResult:
    example_id: str
    condition: str
    cutoff: float
    predicted: Optional[str]
    gold: str
    is_correct: bool
    is_valid: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool
    retry_count: int
    invalid_before_retry: bool
    invalid_after_retry: bool
    fallback_used: bool
    raw_response: str


@dataclass
class ConditionMetrics:
    condition: str
    cutoff: float
    n_examples: int
    accuracy_strict: float
    accuracy_valid_only: float
    invalid_rate: float
    invalid_before_retry_rate: float
    invalid_after_retry_rate: float
    fallback_rate: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_total_tokens: float
    token_reduction_vs_baseline: float
    estimated_cost_usd: float
    actual_cost_usd: float
    delta_strict_vs_baseline: float
    delta_ci_lower: float
    delta_ci_upper: float
    p_better: float
    mcnemar_chi2: float
    mcnemar_pvalue: float


class ResultCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key_path(self, key: CacheKey) -> str:
        name = f"{key.model}_{key.condition}_{key.example_id}_{key.prompt_hash}_{key.retry_index}.json"
        safe = name.replace("/", "_")
        return os.path.join(self.cache_dir, safe)

    def get(self, key: CacheKey) -> Optional[Dict[str, Any]]:
        path = self._key_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: CacheKey, value: Dict[str, Any]) -> None:
        path = self._key_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {"entries": len(os.listdir(self.cache_dir))}
        except Exception:
            return {"entries": 0}


class ArtifactWriter:
    def __init__(self, results_dir: str, run_id: str):
        self.base_dir = os.path.join(results_dir, run_id)
        os.makedirs(self.base_dir, exist_ok=True)

    def _write_text(self, filename: str, text: str) -> None:
        path = os.path.join(self.base_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def write_prompt(self, example_id: str, condition: str, prompt: str, is_baseline: bool = False) -> None:
        tag = "baseline" if is_baseline else condition
        name = f"prompt_{example_id}_{tag}.txt"
        self._write_text(name, prompt)

    def write_response(self, example_id: str, condition: str, response: str, retry_index: int = 0) -> None:
        name = f"response_{example_id}_{condition}_retry{retry_index}.txt"
        self._write_text(name, response)

    def write_parsed_answer(self, example_id: str, condition: str, payload: Dict[str, Any]) -> None:
        name = f"parsed_{example_id}_{condition}.json"
        self._write_text(name, json.dumps(payload, indent=2))


def compute_prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def save_results_json(results_dir: str, config: Dict[str, Any], metrics: List[ConditionMetrics],
                      results: List[ExampleResult], run_metadata: Dict[str, Any]) -> None:
    payload = {
        "config": config,
        "metrics": [asdict(m) for m in metrics],
        "results": [asdict(r) for r in results],
        "metadata": run_metadata,
    }
    path = os.path.join(results_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_results_csv(results_dir: str, metrics: List[ConditionMetrics]) -> None:
    if not metrics:
        return
    path = os.path.join(results_dir, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def save_examples_csv(results_dir: str, results: List[ExampleResult]) -> None:
    if not results:
        return
    path = os.path.join(results_dir, "examples.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_results_table(metrics: List[ConditionMetrics]) -> None:
    if not metrics:
        return
    header = [
        "condition",
        "acc_strict",
        "acc_valid",
        "invalid",
        "mean_in",
        "mean_total",
        "cost_usd",
        "delta",
    ]
    print("\n" + " | ".join(header))
    print("-" * 80)
    for m in metrics:
        row = [
            m.condition,
            f"{m.accuracy_strict:.3f}",
            f"{m.accuracy_valid_only:.3f}",
            f"{m.invalid_rate:.3f}",
            f"{m.mean_input_tokens:.1f}",
            f"{m.mean_total_tokens:.1f}",
            f"{m.actual_cost_usd:.4f}",
            f"{m.delta_strict_vs_baseline:.3f}",
        ]
        print(" | ".join(row))
