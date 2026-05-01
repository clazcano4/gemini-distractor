# Gemini Distractor

Small experiment repo for measuring how short-answer QA performance changes when Gemini is given distracting or misleading context passages.

The pipeline uses SQuAD v1.1 examples, builds several context conditions, runs Gemini on each prompt, evaluates answer accuracy and evidence selection, and generates summary plots.

## Repo Layout

- `scripts/build_dataset.py`: builds the distractor-conditioned evaluation set.
- `scripts/run_experiment_gemini.py`: queries Gemini and saves raw JSONL predictions.
- `scripts/evaluate.py`: computes exact match, F1, evidence attribution, and distractor metrics.
- `scripts/make_plots.py`: renders the figures in `figures/`.
- `data/`: dataset artifacts.
- `outputs/`: local prediction and metric files. Ignored by Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

`GEMINI_API_KEY` is only needed for the model run.

## Run

Build a 100-example evaluation set:

```bash
python3 scripts/build_dataset.py \
  --n 100 \
  --split validation \
  --out data/squad_distractor_eval_100.jsonl
```

Run Gemini on selected conditions:

```bash
python3 scripts/run_experiment_gemini.py \
  --input data/squad_distractor_eval_100.jsonl \
  --output outputs/predictions_gemini.jsonl \
  --model gemini-2.5-flash-lite \
  --prompt_variant robust \
  --conditions relevant_only random_1 random_3 random_5 adversarial_1 misleading_only position_first position_middle position_last \
  --resume
```

Evaluate predictions:

```bash
python3 scripts/evaluate.py \
  --predictions outputs/predictions_gemini.jsonl \
  --out_csv outputs/metrics_gemini.csv \
  --out_errors outputs/errors_gemini.csv
```

Generate plots:

```bash
python3 scripts/make_plots.py \
  --metrics outputs/metrics_gemini.csv \
  --out_dir figures
```

## Conditions

The dataset includes:

- `no_context`
- `relevant_only`
- `random_1`, `random_3`, `random_5`
- `adversarial_1`
- `misleading_only`
- `position_first`, `position_middle`, `position_last`

## Notes

- SQuAD v1.1 is downloaded on demand and cached under `data/cache/`.
- Raw outputs and caches stay local by default via `.gitignore`.
- The committed `figures/` directory contains example visualizations from a completed run.
