# Running the experiments

One script per method of the paper. Every script is plain `bash`, configured by environment variables, and runs
one model on one conflict per call. Run them on any machine with a GPU, or wrap them in your cluster's job
scheduler (each call is one job).

| script | method | paper |
|---|---|---|
| `run_baseline.sh` | Baseline: the original task prompt | Sec. 4.3 (1) |
| `run_neutrality_control.sh` | Neutrality control: an explicit "be neutral" instruction | Sec. 4.3 (2), App. B.2 |
| `run_self_distancing.sh` | Self-distancing (SD) prompting; `VARIANT=v2/v3` = the wording variants | Sec. 3.1, 4.3 (3), App. B.4, I |
| `run_sdas_train.sh` | SDAS 1/3: paired generation + the self-distancing direction (train split) | Sec. 3.2, App. C.1 |
| `run_sdas_validation.sh` | SDAS 2/3: layer screen, then the grid over hand-chosen layers, layer/alpha selection (validation split) | App. C.2-C.3 |
| `run_sdas_test.sh` | SDAS 3/3: held-out evaluation of the selected cell (test split) | Sec. 4.3 (4), 5.1 |
| `run_sdas_transfer.sh` | Cross-conflict transfer of SDAS directions | Sec. 3.2, 5.2 |
| `run_sdas_released.sh` | SDAS test evaluation with a released vector (`SDAS_models/`), no training / validation | Sec. 5.1 |
| `run_sdas_alignment.sh` | Cross-conflict alignment: cosine and norm ratio of the RU-UA vs IL-PS directions | Sec. 3.2, 5.2, Table 2 |
| `run_judge_robustness.sh` | Re-scoring with another judge model or judge prompt | App. D.6-D.7 |
| `run_annotation.sh` | GPT-5.6-Luna mechanism labels (`TASK=mechanisms`) or semantic preservation (`TASK=preservation`) | App. E, G |

`common.sh` holds the shared settings and helpers; it is sourced by the scripts and never run directly.

## Setup

**Python.** Two conda environments, defined in the repository root - see "Environments" in the root
`README.md` for why there are two:

```bash
conda env create -f environment.yml          # sdas: every model except Gemma, all judging, statistics
conda env create -f environment-gemma.yml    # sdas-gemma: only the steps that load Gemma weights
conda activate sdas
```

**Models.** Download the instruction-tuned checkpoints from Hugging Face and pass their local paths:

| results name (`MODEL_NAME`) | model (as named in the paper) | access |
|---|---|---|
| `qwen25-7b` | Qwen2.5-7B-Instruct | local weights (`MODEL_PATH`) |
| `falcon3-7b-instruct` | Falcon3-7B-Instruct | local weights |
| `mistral-7b-instruct-v0.3` | Mistral-7B-Instruct-v0.3 | local weights |
| `aya-expanse-8b` | Aya-Expanse-8B | local weights |
| `gemma-4-e4b-it` | Gemma-4-E4B-IT | local weights (transformers >= 5, see below) |
| `gpt-5.6-luna` | GPT-5.6-Luna | OpenRouter (`REMOTE_MODEL=openai/gpt-5.6-luna REMOTE_REASONING=none REMOTE_TEMPERATURE=none`) |
| `claude-sonnet-5` | Claude-Sonnet-5 | OpenRouter (`REMOTE_MODEL=anthropic/claude-sonnet-5 REMOTE_REASONING=none REMOTE_TEMPERATURE=none`) |
| `kimi-k2.6` | Kimi-K2.6 | OpenRouter (`REMOTE_MODEL=moonshotai/kimi-k2.6 REMOTE_REASONING=none REMOTE_TEMPERATURE=0.4`) |

The judge is **Llama-3.1-8B-Instruct**, always run locally (`JUDGE_PATH`).
Use the `MODEL_NAME` values above: the statistics notebooks and `test_stats.py` find results by these names.

**API key.** Only the closed-weight models need one. Put it in `<repo>/.env`:

```
OPENROUTER_API_KEY=...
```

Optional OpenRouter settings (timeouts, retries, provider pinning) are documented in `src/openrouter/client.py`.

**Gemma-4.** Run every script for `gemma-4-e4b-it` with `PYTHON` set to the `sdas-gemma` environment:

```bash
GEMMA_PY=$(conda run -n sdas-gemma which python)
MAIN_PY=$(conda run -n sdas which python)
PYTHON=$GEMMA_PY TRANSLATOR_DTYPE=float32 JUDGE_DTYPE=float32 MODEL_NAME=gemma-4-e4b-it MODEL_PATH=... bash scripts/run_baseline.sh
PYTHON=$GEMMA_PY JUDGE_PYTHON=$MAIN_PY MODEL_NAME=gemma-4-e4b-it MODEL_PATH=... bash scripts/run_sdas_train.sh
PYTHON=$GEMMA_PY JUDGE_PYTHON=$MAIN_PY MODEL_NAME=gemma-4-e4b-it bash scripts/run_sdas_validation.sh
PYTHON=$GEMMA_PY JUDGE_PYTHON=$MAIN_PY LAYERS=13,18,20,22 MODEL_NAME=gemma-4-e4b-it bash scripts/run_sdas_validation.sh  # grid
```

`run_sdas_test.sh` and `run_sdas_transfer.sh` take the same `PYTHON` / `JUDGE_PYTHON` pair; `run_judge_robustness.sh`
loads no generator and runs entirely in `sdas`. For Gemma, choose candidate layers below 24: the last 18 of
Gemma's 42 layers reuse KV computed by earlier layers, so steering there is a different intervention.

## Data

`data/ru_ua/` (Russia-Ukraine; target languages Russian `ru`, Ukrainian `ua`) and `data/il_ps/`
(Israel-Palestine; Arabic `ps`, Hebrew `il`), each split into `neutral_data_{train,val,test}.json` with 100
events per split. Each item holds the neutral English event statement and the two side-oriented reference
snippets (the narrative anchors). Select the conflict with `PAIR=ru_ua` or `PAIR=il_ps`. The perspectival gap is
`d_UA - d_RU` (RU-UA) and `d_IL - d_PS` (IL-PS).

The prompting conditions are evaluated on the test split. SDAS learns its direction on train, selects its layer and
alpha on val, and is evaluated on test.

## Running everything for one model

```bash
export JUDGE_PATH=/path/to/Llama-3.1-8B-Instruct
export MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

for PAIR in ru_ua il_ps; do
  export PAIR
  bash scripts/run_baseline.sh
  bash scripts/run_neutrality_control.sh
  bash scripts/run_self_distancing.sh
  VARIANT=v2 bash scripts/run_self_distancing.sh
  VARIANT=v3 bash scripts/run_self_distancing.sh
  bash scripts/run_sdas_train.sh
  bash scripts/run_sdas_validation.sh
done

PAIR=ru_ua LAYERS=11,12,16,17 bash scripts/run_sdas_validation.sh
PAIR=ru_ua CELL=L17_a1 bash scripts/run_sdas_test.sh
PAIR=il_ps LAYERS=12,16,17 bash scripts/run_sdas_validation.sh
PAIR=il_ps CELL=L16_a1 bash scripts/run_sdas_test.sh

TRANSFER=ru_ua_to_il_ps CELL=L17_a1 bash scripts/run_sdas_transfer.sh
TRANSFER=il_ps_to_ru_ua CELL=L16_a1 bash scripts/run_sdas_transfer.sh

MODELS="qwen25-7b:17,16" bash scripts/run_sdas_alignment.sh
```

**Judge robustness**, once the outputs of all models exist (one call per judge configuration and conflict; every
stored output of every condition is re-scored, nothing is regenerated):

```bash
for PAIR in ru_ua il_ps; do
  export PAIR
  JUDGE_PROMPT=EVENT_DISTANCE_JUDGE bash scripts/run_judge_robustness.sh                  # judge prompt (App. D.6)
  JUDGE_NAME=falcon3-7b-instruct JUDGE_PATH=/path/to/Falcon3-7B-Instruct \
    bash scripts/run_judge_robustness.sh                                                  # judge model, local (App. D.7)
  JUDGE_NAME=gpt-5.1 JUDGE_PATH= REMOTE_JUDGE=openai/gpt-5.1 bash scripts/run_judge_robustness.sh   # judge model, API
done
```

The re-scored runs sit next to the primary ones under `<condition>/<JUDGE_NAME>_judge/<JUDGE_PROMPT>/`. The
agreement with the primary judge (Cohen's κ, App. D.7) is computed in `src/statistics/calculate_kappa.ipynb`; the
intervention effects under another judge come from `test_stats.py --judge <JUDGE_NAME> --judge_prompt <JUDGE_PROMPT>`.

The closed-weight models run only the three prompting scripts (no activation access), e.g.
`MODEL_NAME=gpt-5.6-luna REMOTE_MODEL=openai/gpt-5.6-luna REMOTE_REASONING=none REMOTE_TEMPERATURE=none bash scripts/run_baseline.sh`.

The candidate layers chosen from the screens (`LAYERS`) and the cells selected on validation in the paper
(App. C.3, Tables 9 and 11):

| model | RU-UA `LAYERS` | RU-UA cell | IL-PS `LAYERS` | IL-PS cell |
|---|---|---|---|---|
| qwen25-7b | `11,12,16,17` | `L17_a1` | `12,16,17` | `L16_a1` |
| falcon3-7b-instruct | `7,14,16` | `L16_a2` | `8,14,16` | `L16_a2` |
| mistral-7b-instruct-v0.3 | `8,16,19` | `L16_a2` | `8,16,18` | `L16_a2` |
| aya-expanse-8b | `14,16,18` | `L14_a2` | `14,15,18` | `L14_a2` |
| gemma-4-e4b-it | `13,18,20,22` | `L13_a2` | `12,13,18,20` | `L12_a2` |

The RU-UA directions at these layers are released in `SDAS_models/` (see its README), so SDAS can be applied
without retraining: `run_sdas_released.sh` runs the test evaluation with them, and
`src/pipeline/sdas/sdas_generate.py` generates with one for any prompt.

**Before a full run**, check the setup cheaply: `DRY_RUN=1` prints every command without running anything, and
`LIMIT=2` processes only the first 2 events of each data file.

All steps are resumable: a step that is re-run skips the generations and judgements that already exist.

## Settings

Shared by every script (`common.sh`):

| variable | meaning | default |
|---|---|---|
| `PAIR` | `ru_ua` or `il_ps` | `ru_ua` |
| `PYTHON` | Python interpreter | `python` |
| `JUDGE_PYTHON` | interpreter for judging / statistics steps (Gemma: the `sdas` environment) | `PYTHON` |
| `JUDGE_PATH` | local weights of the judge | required |
| `JUDGE_NAME` | judge results-bucket name | `llama3-8b` |
| `JUDGE_PROMPT` | judge prompt (`src/judge/prompts/prompts.py`) | `BASE_JUDGE` |
| `LIMIT` | first N events per data file (smoke tests) | all |
| `DRY_RUN` | `1` = print commands only | `0` |

Prompting scripts: `MODEL_NAME`, and either `MODEL_PATH` (local) or `REMOTE_MODEL` (+ `REMOTE_REASONING`,
`REMOTE_TEMPERATURE`); `TRANSLATOR_DTYPE`, `JUDGE_DTYPE`; `VARIANT` for `run_self_distancing.sh`.
SDAS scripts: `MODEL_NAME`, `MODEL_PATH`, `INTERVENTION`, plus `LAYERS`, `ALPHAS`,
`SCREEN_N_PAIRS` (validation), `CELL` (test, transfer) and `TRANSFER` (transfer).
Judge robustness: `JUDGE_NAME`, `JUDGE_PROMPT` and `JUDGE_PATH` or `REMOTE_JUDGE`, `TREES`, `MODELS`. Each script's header documents its
settings, and every Python entry point has `--help`.

## Outputs

Everything is written under `results/` (RU-UA) or `results_il_ps/` (IL-PS):

```
results[_il_ps]/
├── baseline/ control/ self_distance/          prompting conditions
│   └── <judge>_judge/<judge_prompt>/<model>/<prompt>/<lang>_run_<timestamp>/
│       ├── <lang>_results.jsonl              one row per event: output, judge score, explanation
│       └── <lang>_summary.json
├── self_distance_ablation/self_distance_v{2,3}/…   wording variants, same layout
├── sdas/<model>/                          SDAS experiment: generations + activations, directions,
│                                              screens, judged val cells, val summaries
├── sdas_test/…/<model>_<cell>/…           SDAS test runs, in the layout of the prompting conditions
└── cross_conflict/                            transfer experiments and their test runs
```

## Mechanism labels and semantic preservation

`TASK=mechanisms|preservation bash scripts/run_annotation.sh` (settings: `ANNOTATOR`, `TREES`, `MODELS`). Prompts:
`src/annotation/prompts.py`. Output: `results[_il_ps]/annotations/<task>/`, with `summary_<langs>.csv`.

## Statistics

- `src/pipeline/sdas/test_stats.py` compares every condition with the baseline on the test split: gap, change
  in |gap|, event-clustered bootstrap CI (2,000 resamples), event-level permutation test (5,000 permutations) and
  Benjamini-Hochberg q. `run_sdas_test.sh` and `run_sdas_transfer.sh` call it. For the neutrality control, run
  `python src/pipeline/sdas/test_stats.py --pair <pair> --injection control --include_injection`.
- `src/statistics/*.ipynb` produce the paper's tables: the main comparison (`calculate_gap_test_sdas.ipynb`,
  `calculate_gap_test_sdas_vs_self_distance.ipynb`), the inter-judge agreement (`calculate_kappa.ipynb`) and the
  wording robustness (`self_distance_ablations/`). The gap notebooks cover both conflicts: set `PAIR` in their first cell.
