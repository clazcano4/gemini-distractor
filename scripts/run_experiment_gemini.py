#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from google import genai
from google.genai import types


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def already_completed_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "run_id" in row and not row.get("api_error"):
                    done.add(row["run_id"])
            except json.JSONDecodeError:
                continue
    return done


def build_prompt(row: Dict, prompt_variant: str) -> str:
    question = row["question"]
    condition = row["condition"]

    if condition == "no_context":
        return f"""Answer the question with the shortest possible answer string.
Do not answer in a full sentence.
Do not include explanation.
Return only valid JSON with keys "answer" and "evidence_id".
For this no-context condition, set "evidence_id" to null.
Do not include markdown, code fences, explanations, or extra text.

Question:
{question}

Return only JSON:
"""

    context = row["context"]

    if prompt_variant == "robust":
        instruction = (
            'You are answering a short-answer question using the numbered context passages below. '
            'Some passages may be irrelevant or misleading. Use only passages that directly support the answer. '
            'If passages conflict, choose the answer supported by the most explicit relevant evidence. '
            'Return the shortest answer string that directly answers the question. '
            'Prefer copying the exact answer span from the relevant passage when possible. '
            'Do not answer in a full sentence. '
            'Do not include explanation. '
            'Return only valid JSON with keys "answer" and "evidence_id". '
            'The evidence_id must be the number of the context passage that supports the answer. '
            'Do not include markdown, code fences, explanations, or extra text.'
        )
    else:
        instruction = (
            'You are answering a short-answer question using the numbered context passages below. '
            'Return the shortest answer string that directly answers the question. '
            'Prefer copying the exact answer span from the relevant passage when possible. '
            'Do not answer in a full sentence. '
            'Do not include explanation. '
            'Return only valid JSON with keys "answer" and "evidence_id". '
            'The evidence_id must be the number of the context passage that supports the answer. '
            'Do not include markdown, code fences, explanations, or extra text.'
        )

    return f"""{instruction}

Context:
{context}

Question:
{question}

Return only JSON:
"""


def call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    force_json: bool,
) -> str:
    if force_json:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )
    else:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--prompt_variant", default="basic", choices=["basic", "robust"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=6.0, help="Seconds between calls to respect free-tier limits")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=30.0)
    parser.add_argument("--no_force_json", action="store_true", help="Disable response_mime_type=application/json")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY before running this script.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    client = genai.Client()

    completed = already_completed_ids(args.output) if args.resume else set()
    condition_set = set(args.conditions)
    rows = [row for row in load_jsonl(args.input) if row.get("condition") in condition_set]
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"Running {total} rows")
    print(f"Model: {args.model}")
    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}")
    print("Cost guardrail: keep Google AI Studio billing disabled; use --resume if rate-limited.")

    for idx, row in enumerate(rows, start=1):
        run_id = f'{row["id"]}::{args.prompt_variant}::{args.model}'
        if run_id in completed:
            continue

        prompt = build_prompt(row, args.prompt_variant)
        raw_response = ""
        error: Optional[str] = None

        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = call_gemini(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    force_json=not args.no_force_json,
                )
                error = None
                break
            except Exception as e:
                error = repr(e)
                print(f"Error on row {idx}/{total}, attempt {attempt}/{args.max_retries}: {error}")
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep)

        out = dict(row)
        out.update(
            {
                "run_id": run_id,
                "model": args.model,
                "prompt_variant": args.prompt_variant,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "raw_response": raw_response,
                "api_error": error,
            }
        )
        append_jsonl(args.output, out)

        if idx % 10 == 0 or idx == total:
            print(f"Completed {idx}/{total}")

        time.sleep(args.sleep)

    print(f"Done. Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
