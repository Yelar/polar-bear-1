"""
Prompt building and validation utilities for LongBench v2 evaluation.
"""

import re
from typing import Dict, Tuple, Any, List, Optional

FINAL_INSTRUCTION = "Answer with A/B/C/D."


def build_prompt_from_example(example: Dict[str, Any], context: str) -> str:
    """Build a prompt from a dataset example and context."""
    question = (
        example.get("question")
        or example.get("input")
        or example.get("task")
        or example.get("prompt")
        or ""
    )
    choices = (
        example.get("choices")
        or example.get("options")
        or example.get("candidates")
        or example.get("candidate_answers")
    )

    parts: List[str] = []
    if question:
        parts.append(question.strip())

    if context:
        parts.append("Context:\n" + context.strip())

    if choices:
        parts.append("Options:\n" + _format_choices(choices))

    parts.append(FINAL_INSTRUCTION)
    return "\n\n".join([p for p in parts if p])


def parse_answer(text: str) -> Optional[str]:
    """Parse A/B/C/D answer from model output."""
    if not text:
        return None
    match = re.search(r"answer\s*[:\-]?\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()
    return None


def build_repair_prompt(raw_response: str) -> str:
    """Prompt to repair invalid model output."""
    return (
        "Your previous answer was invalid or not in the expected format. "
        "Reply with a single letter A, B, C, or D only."
    )


def validate_compressed_prompt(prompt: str, example: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Validate compressed prompt structure before sending to model."""
    details: Dict[str, Any] = {
        "empty": False,
        "unbalanced_code_fences": False,
        "missing_keywords": False,
    }
    if not prompt or not prompt.strip():
        details["empty"] = True
        return False, details

    if prompt.count("```") % 2 == 1:
        details["unbalanced_code_fences"] = True
        return False, details

    question = (
        example.get("question")
        or example.get("input")
        or example.get("task")
        or example.get("prompt")
        or ""
    )
    if question:
        keywords = _extract_keywords(question)
        if keywords and not any(kw in prompt.lower() for kw in keywords):
            details["missing_keywords"] = True
            return False, details

    return True, details


def ensure_final_instruction(prompt: str) -> str:
    """Ensure the final instruction is present at the end of the prompt."""
    if FINAL_INSTRUCTION.lower() in prompt.lower():
        return prompt
    return prompt.rstrip() + "\n\n" + FINAL_INSTRUCTION


def _extract_keywords(text: str) -> List[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
        "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
        "were", "be", "by", "from", "that", "this", "it", "as", "at", "if", "then",
        "than", "into", "about", "over", "under", "between", "within", "without",
    }
    tokens = re.findall(r"[A-Za-z0-9_\-/]+", text.lower())
    keywords = [tok for tok in tokens if tok not in stopwords and len(tok) > 2]
    return list(dict.fromkeys(keywords))


def _format_choices(choices: Any) -> str:
    if isinstance(choices, dict):
        lines = []
        for key in sorted(choices.keys()):
            lines.append(f"{key}. {choices[key]}")
        return "\n".join(lines)
    if isinstance(choices, (list, tuple)):
        lines = []
        letters = ["A", "B", "C", "D"]
        for idx, option in enumerate(choices):
            prefix = letters[idx] if idx < len(letters) else str(idx + 1)
            lines.append(f"{prefix}. {option}")
        return "\n".join(lines)
    return str(choices)
