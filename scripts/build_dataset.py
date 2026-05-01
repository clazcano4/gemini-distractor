#!/usr/bin/env python3

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

SQUAD_URLS = {
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
}

CONDITIONS = [
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


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def answer_token_len(answer: str) -> int:
    return len(normalize_text(answer).split())


def format_context(passages: List[str]) -> str:
    blocks = []
    for i, passage in enumerate(passages, start=1):
        blocks.append(f"Context {i}:\n{normalize_text(passage)}")
    return "\n\n".join(blocks)


def load_squad(split: str, cache_dir: str) -> List[Dict]:
    """Download/cache SQuAD v1.1 and flatten into QA rows."""
    if split not in SQUAD_URLS:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {sorted(SQUAD_URLS)}")

    cache_path = Path(cache_dir) / f"squad_v1_1_{split}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        url = SQUAD_URLS[split]
        print(f"Downloading SQuAD {split} split from {url}")
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as e:
            raise RuntimeError(
                "Could not download SQuAD. Check your internet connection, or manually download "
                f"{url} and save it as {cache_path}"
            ) from e
    else:
        print(f"Using cached SQuAD file: {cache_path}")

    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for article in raw.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers_text = [a.get("text", "") for a in qa.get("answers", []) if a.get("text")]
                if not answers_text:
                    continue
                rows.append({
                    "id": qa.get("id", ""),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": {"text": answers_text},
                })
    return rows


def valid_item(item: Dict, max_answer_tokens: int) -> bool:
    answers = item.get("answers", {}).get("text", [])
    if not answers:
        return False
    first_answer = normalize_text(answers[0])
    context = item.get("context", "")
    if not first_answer or answer_token_len(first_answer) > max_answer_tokens:
        return False
    return first_answer.lower() in context.lower()


def sample_wrong_answer(pool: List[Dict], gold_answers: List[str], rng: random.Random) -> str:
    gold_norm = {a.lower().strip() for a in gold_answers}
    for _ in range(500):
        cand_item = rng.choice(pool)
        cand_answers = cand_item.get("answers", {}).get("text", [])
        if not cand_answers:
            continue
        cand = normalize_text(cand_answers[0])
        if cand and cand.lower() not in gold_norm and answer_token_len(cand) <= 8:
            return cand
    return "an unrelated entity"


def make_adversarial_passage(context: str, gold_answer: str, wrong_answer: str) -> str:
    pattern = re.compile(re.escape(gold_answer), flags=re.IGNORECASE)
    return pattern.sub(wrong_answer, context, count=1)


def sample_distractors(pool: List[Dict], base_id: str, k: int, rng: random.Random) -> List[str]:
    distractors = []
    seen_contexts = set()
    attempts = 0
    while len(distractors) < k and attempts < 5000:
        attempts += 1
        cand = rng.choice(pool)
        if str(cand.get("id")) == str(base_id):
            continue
        ctx = normalize_text(cand.get("context", ""))
        if not ctx or ctx in seen_contexts:
            continue
        seen_contexts.add(ctx)
        distractors.append(ctx)
    if len(distractors) < k:
        raise RuntimeError(f"Could not sample {k} distractors")
    return distractors


def shuffled_with_gold(gold_context: str, distractors: List[str], rng: random.Random) -> Tuple[List[str], int]:
    passages = distractors + [gold_context]
    rng.shuffle(passages)
    gold_id = passages.index(gold_context) + 1
    return passages, gold_id


def build_rows_for_item(item: Dict, pool: List[Dict], rng: random.Random) -> List[Dict]:
    base_id = str(item["id"])
    title = item.get("title", "")
    question = normalize_text(item["question"])
    gold_context = normalize_text(item["context"])
    gold_answers = [normalize_text(a) for a in item["answers"]["text"] if normalize_text(a)]
    primary_answer = gold_answers[0]
    wrong_answer = sample_wrong_answer(pool, gold_answers, rng)
    adversarial_context = make_adversarial_passage(gold_context, primary_answer, wrong_answer)

    rows = []

    def base_row(condition: str) -> Dict:
        return {
            "id": f"{base_id}::{condition}",
            "base_id": base_id,
            "title": title,
            "question": question,
            "gold_answers": gold_answers,
            "condition": condition,
            "context": "",
            "gold_evidence_id": None,
            "adversarial_evidence_id": None,
            "adversarial_answer": wrong_answer,
            "num_passages": 0,
        }

    row = base_row("no_context")
    rows.append(row)

    row = base_row("relevant_only")
    row.update({"context": format_context([gold_context]), "gold_evidence_id": 1, "num_passages": 1})
    rows.append(row)

    for k in [1, 3, 5]:
        condition = f"random_{k}"
        distractors = sample_distractors(pool, base_id, k, rng)
        passages, gold_id = shuffled_with_gold(gold_context, distractors, rng)
        row = base_row(condition)
        row.update({"context": format_context(passages), "gold_evidence_id": gold_id, "num_passages": len(passages)})
        rows.append(row)

    passages = [gold_context, adversarial_context]
    rng.shuffle(passages)
    row = base_row("adversarial_1")
    row.update({
        "context": format_context(passages),
        "gold_evidence_id": passages.index(gold_context) + 1,
        "adversarial_evidence_id": passages.index(adversarial_context) + 1,
        "num_passages": len(passages),
    })
    rows.append(row)

    row = base_row("misleading_only")
    row.update({"context": format_context([adversarial_context]), "adversarial_evidence_id": 1, "num_passages": 1})
    rows.append(row)

    for condition, gold_index in {"position_first": 0, "position_middle": 2, "position_last": 4}.items():
        distractors = sample_distractors(pool, base_id, 4, rng)
        passages = distractors[:]
        passages.insert(gold_index, gold_context)
        row = base_row(condition)
        row.update({"context": format_context(passages), "gold_evidence_id": gold_index + 1, "num_passages": len(passages)})
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of base QA examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--max_answer_tokens", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = load_squad(args.split, args.cache_dir)
    valid = [x for x in pool if valid_item(x, args.max_answer_tokens)]
    rng.shuffle(valid)

    if args.n > len(valid):
        raise ValueError(f"Requested n={args.n}, but only {len(valid)} valid examples are available")

    rows = []
    for item in valid[: args.n]:
        rows.extend(build_rows_for_item(item, valid, rng))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Base examples: {args.n}")
    print(f"Conditions: {', '.join(CONDITIONS)}")


if __name__ == "__main__":
    main()
