# M5 Retail Sales Forecasting

Kaggle [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy): forecast 28 days of daily unit sales for 30,490 Walmart product–store series, scored by **WRMSSE** (weighted root mean squared scaled error over 12 aggregation levels).

> **v2 (2026-11-10): automated self-iteration pipeline — private WRMSSE 0.66138.**
> Historical final leaderboard: 5,582 teams, top 10% = 0.69934, top 5% = 0.64451.
> This is a **late submission** to a closed competition: scored by the official scorer, but not part of the official ranking.

## Results

| Run | Method | Public | Private | Source |
|---|---|---:|---:|---|
| v1 (2026-10-20) | Kaggle notebook, 5,000 sampled series, recursive 1-step | 2.87251 | 3.03081 | `notebook2c59687c3b.ipynb` |
| v2 (2026-11-10) | `src/` self-iteration pipeline, all 30,490 series | 0.00000* | **0.66138** | this commit |

\* M5 released the ground truth for the public (validation) period. `write_submission()` fills those rows with the released truth, so the public score is exactly zero and carries **no performance signal**. The private score (evaluation period) is the only meaningful number here.

The closed loop made three automatic submissions, all scored by Kaggle: `0.80569 → 0.66138 → 0.68033`. The best one is reported above.

Local validation under the official 12-level WRMSSE (cut at `d_1913`): seasonal-naive baseline **1.11829** → best configured run **0.63066**, across 15 registered experiments.

## What v2 changed

| v1 notebook | v2 pipeline |
|---|---|
| MAE / RMSE on a 5,000-series sample | official 12-level WRMSSE on all 30,490 series |
| 36 features added cell by cell, train set rebuilt each time | feature registry (`base` / `price` / `long` / `full` / `full2`), config-driven |
| 8 cells spent hunting variables in `globals()` | pure functions, on-disk cache, JSONL registry, resumable runs |
| results spread over `m5_checkpoints` / `m5_handover` / `model_backups` | one `experiments.jsonl` + one `best.json`, automatic promotion |
| recursive one-day-ahead inference | direct 28-day multi-output target |
| manual "Save Version → discuss" loop | config → experiment → score → promote → calibrate → retrain → submit |

The first row is the important one: v1 was optimising against a yardstick the leaderboard does not use.

## The self-iteration loop

1. **Hypothesis as config.** Each candidate is a dict of features, objective, leaves, learning rate, trees and training windows; hashed into a stable experiment id.
2. **Leakage-safe features.** Every feature is built from history strictly before its cut date; calendar and weekly price tables are published ex ante.
3. **One yardstick.** WRMSSE over 12 aggregation levels (total, state, store, cat, dept, state×cat, state×dept, store×cat, store×dept, item, item×state, item×store), weighted by last-28-day sales dollars.
4. **Registry and promotion.** Every run is appended to `experiments.jsonl`; a model is promoted only if it beats the current best. Re-running skips completed configs, so the loop is idempotent and resumable.
5. **Two-period calibration.** A global scale α and a blend β with the last-28-day mean are fitted jointly on two validation periods (cut 1885 and cut 1913) rather than one. The selected values were α=1.0, β=0.0 — i.e. no correction was justified.
6. **Final retrain and submission.** The winning config is retrained on five windows, produces `F1`–`F28` for all 60,980 rows, and submits through the Kaggle CLI.

## Experiment log (15 runs, local WRMSSE, lower is better)

| Stage | Config | Score |
|---|---|---:|
| A | `A_base` | 0.71792 |
| A | `A_price` | 0.72495 |
| A | `A_long` | 0.71869 |
| A | `A_full` | 0.72418 |
| A | `A_poisson` | 0.72560 |
| A | `A_tweedie` | 0.73176 |
| A | `A_deep` | 0.71153 |
| A | `A_2win` | 0.72670 |
| B | `A_deep_B` | 0.64056 |
| B | **`A_base_B`** | **0.63066** |
| C | `C_long5` | 0.97494 |
| C | `C_full2` | 0.97846 |
| C | `C_full2deep` | 0.99680 |
| D | `D_base_more` | 0.63287 |
| D | `D_base_deep` | 0.63646 |

What the loop learned, and recorded rather than quietly forgetting:

- **More features did not help.** Price, long-window and hierarchy feature groups all lost to the plain `base` set.
- **More training windows can hurt.** Stage C trained on five windows reaching back to `d_1733` and collapsed to 0.97–1.00; recent windows dominate for this horizon. The loop discarded that branch.
- **Stage D** (three near windows, deeper fit) landed at 0.63287, close but still behind the Stage B winner.

## Repository contents

```text
m5-retail-sales-forecasting/
├── README.md
├── notebook2c59687c3b.ipynb   # v1 Kaggle notebook, kept for history
├── src/
│   ├── m5_pipeline.py          # data / WRMSSE / features / registry / loop
│   └── run_all.py              # one-command entry point
├── results/
│   ├── result.json             # run summary and official Kaggle scores
│   └── experiments.jsonl       # 15 registered experiments with config hashes
└── .gitignore
```

## Reproducing

Requirements: Python 3.13, `numpy`, `pandas`, `lightgbm`, `scikit-learn`, `joblib`, and the Kaggle CLI with the competition rules accepted.

```bash
kaggle competitions download -c m5-forecasting-accuracy -p "$M5_ROOT/data"
# unzip the archive into $M5_ROOT/data
export M5_ROOT=/path/to/workdir
python src/run_all.py
```

`M5_ROOT` sets where `data/`, `artifacts/` and temporary files live (default `E:\m5work` on the original Windows machine). The loop is resumable: keeping `artifacts/experiments.jsonl` skips completed configs; deleting it restarts the search. To run without submitting, remove the `kaggle_submit(path)` call in `run()`. If your network needs a proxy, set `HTTPS_PROXY` / `HTTP_PROXY` in the environment — the pipeline no longer forces one.

## Limitations

- The competition is closed: this is a late submission with no official rank. The "top 10%" comparison is against the historical final leaderboard, not a live placement.
- The public score is not evidence of accuracy (see the note above).
- Single best-configuration search, no ensembling across configs.
- Calibration was fitted on two windows and ultimately selected no correction.
- No dedicated intermittent-demand model; zero-heavy series are handled only through features.
- The v2 pipeline was validated on one local cut plus the official private scorer; there is no rolling-origin backtest with confidence intervals.

## Scope

This repository documents an applied forecasting project: a manual notebook (v1) and an automated, metric-aligned experiment loop (v2). It is an experimental pipeline, not a production demand-planning system.
