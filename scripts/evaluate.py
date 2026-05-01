#!/usr/bin/env python3

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match(prediction: str, gold_answers: List[str]) -> int:
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(gold) for gold in gold_answers))


def f1_score_single(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, gold_answers: List[str]) -> float:
    return max((f1_score_single(prediction, gold) for gold in gold_answers), default=0.0)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove first and last code fence if present.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_response(raw: str) -> Tuple[Optional[str], Optional[int], bool, str]:
    """Return answer, evidence_id, parse_ok, parse_error."""
    if raw is None:
        return None, None, False, "empty_response"
    text = strip_code_fences(str(raw))
    if not text:
        return None, None, False, "empty_response"

    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, None, False, "no_json_object"
        try:
            obj = json.loads(match.group(0))
        except Exception as e:
            return None, None, False, f"json_parse_error:{e}"

    if not isinstance(obj, dict):
        return None, None, False, "json_not_object"

    answer = obj.get("answer", None)
    evidence = obj.get("evidence_id", None)

    evidence_id: Optional[int]
    if evidence is None or str(evidence).strip().lower() in {"null", "none", "n/a", "na", ""}:
        evidence_id = None
    else:
        try:
            evidence_id = int(evidence)
        except Exception:
            evidence_id = None

    if answer is None:
        return None, evidence_id, False, "missing_answer"

    return str(answer), evidence_id, True, ""


def copied_adversarial(prediction: str, adversarial_answer: Optional[str]) -> int:
    if not adversarial_answer:
        return 0
    return int(normalize_answer(prediction) == normalize_answer(adversarial_answer))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_errors", required=True)
    args = parser.parse_args()

    evaluated_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for row in load_jsonl(args.predictions):
        raw_response = row.get("raw_response", "")
        answer, evidence_id, parse_ok, parse_error = parse_response(raw_response)
        gold_answers = row.get("gold_answers", [])
        if isinstance(gold_answers, str):
            try:
                gold_answers = json.loads(gold_answers)
            except Exception:
                gold_answers = [gold_answers]

        api_error = row.get("api_error")
        prediction = answer or ""
        acc = exact_match(prediction, gold_answers) if parse_ok and not api_error else 0
        f1 = max_f1(prediction, gold_answers) if parse_ok and not api_error else 0.0

        gold_evidence_id = row.get("gold_evidence_id")
        has_gold_evidence = gold_evidence_id is not None
        evidence_correct: Optional[int]
        if has_gold_evidence and parse_ok and not api_error:
            evidence_correct = int(evidence_id == int(gold_evidence_id))
        else:
            evidence_correct = None

        adversarial_answer = row.get("adversarial_answer")
        copied_adv = copied_adversarial(prediction, adversarial_answer) if parse_ok and not api_error else 0

        eval_row = dict(row)
        eval_row.update(
            {
                "prediction": prediction,
                "predicted_evidence_id": evidence_id,
                "parse_ok": int(parse_ok),
                "parse_error": parse_error,
                "exact_match": acc,
                "f1": f1,
                "evidence_correct": evidence_correct,
                "copied_adversarial_answer": copied_adv,
                "api_error_flag": int(bool(api_error)),
            }
        )
        evaluated_rows.append(eval_row)

        if (not parse_ok) or api_error or acc == 0 or (evidence_correct == 0):
            error_rows.append(eval_row)

    df = pd.DataFrame(evaluated_rows)
    if df.empty:
        raise ValueError("No predictions found")

    metrics = []
    relevant_only_acc = None
    if (df["condition"] == "relevant_only").any():
        relevant_only_acc = float(df[df["condition"] == "relevant_only"]["exact_match"].mean())

    order = [
        "no_context",
        "relevant_only",
        "random_1",
        "random_3",
        "random_5",
        "adversarial_1",
        "misleading_only",
        "position_first",
        "position_middle",
        "position_last",
    ]

    for condition, group in df.groupby("condition"):
        n = len(group)
        accuracy = float(group["exact_match"].mean())
        f1 = float(group["f1"].mean())
        parse_rate = float(group["parse_ok"].mean())
        api_error_rate = float(group["api_error_flag"].mean())

        evidence_values = group["evidence_correct"].dropna()
        evidence_acc = float(evidence_values.mean()) if len(evidence_values) else None
      
        mcs = float(group["copied_adversarial_answer"].mean()) if "adversarial_answer" in group.columns else None

        dss = None
        if relevant_only_acc is not None:
            dss = relevant_only_acc - accuracy

        metrics.append(
            {
                "condition": condition,
                "n": n,
                "accuracy": accuracy,
                "f1": f1,
                "evidence_attribution_accuracy": evidence_acc,
                "misleading_context_susceptibility": mcs,
                "distractor_sensitivity": dss,
                "parse_rate": parse_rate,
                "api_error_rate": api_error_rate,
            }
        )

    mdf = pd.DataFrame(metrics)
    mdf["condition"] = pd.Categorical(mdf["condition"], order, ordered=True)
    mdf = mdf.sort_values("condition")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_errors).parent.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(args.out_csv, index=False)
    pd.DataFrame(error_rows).to_csv(args.out_errors, index=False)

    print(f"Wrote metrics to {args.out_csv}")
    print(f"Wrote errors to {args.out_errors}")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
