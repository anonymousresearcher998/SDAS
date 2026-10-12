# Steering Away from Conflict: Self-Distancing Activation Steering for Mitigating Perspectival Divergence in LLMs

Code, data and steering vectors for the paper (anonymous NAACL 2027 submission).

- **Self-distancing (SD)**: a prompting intervention (Kross et al., 2011). Reduces perspectival divergence by 45% on
  average across two conflicts, four languages and eight models, preserving the event core in 93% of outputs.
- **Self-Distancing Activation Steering (SDAS)**: turns the shift SD induces into a steering direction added at one
  layer, with the original prompt. Reduces divergence by 58% across five open-weight models, preserving 91%.

Section and table numbers refer to the paper. Script settings and outputs: [`scripts/README.md`](scripts/README.md).

## Repository layout

| path | contents |
|---|---|
| `data/` | `ru_ua/` (Russia-Ukraine) and `il_ps/` (Israel-Palestine) |
| `scripts/` | one `bash` script per method, `common.sh` (shared settings) |
| `src/translator/` | generation; the 13 communication-context prompts (`prompts/prompts.py`, App. B.3) |
| `src/mitigation/` | SD1/SD2/SD3 (`psychological_interventions/self_distance.py`, App. B.4); neutrality control (`control.py`, App. B.2) |
| `src/judge/` | semantic-distance judge and its prompts (`prompts/prompts.py`, App. D.2, D.6) |
| `src/pipeline/` | prompting pipeline (`en_to_{ru,ua,il,ps}_bias.py`), judge re-scoring (`en_judge_same.py`), annotation (`annotate.py`) |
| `src/sdas/`, `src/pipeline/sdas/` | SDAS internals (steering hook: `hooked_model.py`) and stages |
| `src/annotation/` | mechanism-labeler (App. G.2) and semantic-preservation (App. E) prompts |
| `src/statistics/` | notebooks for the paper's tables |
| `SDAS_models/` | released SDAS vectors ([`SDAS_models/README.md`](SDAS_models/README.md)) |

## Data

300 events per conflict, split 100 / 100 / 100 (`data/<pair>/neutral_data_{train,val,test}.json`, Sec. 4.1, App. A).
Each event: a neutral English statement (the model input) and two side-oriented narrative anchors.

| conflict | `PAIR` | languages | gap in the code |
|---|---|---|---|
| Russia-Ukraine | `ru_ua` | `ru`, `ua` | `d_UA - d_RU` |
| Israel-Palestine | `il_ps` | `ps` (Arabic), `il` (Hebrew) | `d_IL - d_PS` |

Prompting conditions use test. SDAS learns on train, selects on val, evaluates on test.

## Models

| paper | model | `MODEL_NAME` | access |
|---|---|---|---|
| Qwen | Qwen2.5-7B-Instruct | `qwen25-7b` | local |
| Falcon | Falcon3-7B-Instruct | `falcon3-7b-instruct` | local |
| Mistral | Mistral-7B-Instruct-v0.3 | `mistral-7b-instruct-v0.3` | local |
| Aya | Aya-Expanse-8B | `aya-expanse-8b` | local |
| Gemma | Gemma-4-E4B-IT | `gemma-4-e4b-it` | local (`sdas-gemma` env) |
| GPT | GPT-5.6-Luna | `gpt-5.6-luna` | OpenRouter |
| Kimi | Kimi-K2.6 | `kimi-k2.6` | OpenRouter |
| Claude | Claude-Sonnet-5 | `claude-sonnet-5` | OpenRouter |

SDAS runs only on the five local models. Judges: Llama-3.1-8B-Instruct (primary, local); Falcon3-7B-Instruct and
GPT-5.1 (robustness, App. D.7). Annotator: GPT-5.6-Luna, high reasoning (App. E, G).

## Setup

```bash
conda env create -f environment.yml          # sdas: transformers 4.55.2, everything except Gemma
conda env create -f environment-gemma.yml    # sdas-gemma: transformers 5.16.1, Gemma generation and steering only
conda activate sdas
export JUDGE_PATH=/path/to/Llama-3.1-8B-Instruct
```

Gemma-4 needs transformers 5; the reported numbers for the rest use 4.55.2. Each SDAS experiment records its
transformers major version and refuses to continue under the other. For Gemma, set `PYTHON` to `sdas-gemma` and
`JUDGE_PYTHON` to `sdas` (see [`scripts/README.md`](scripts/README.md)).

API models: put `OPENROUTER_API_KEY=...` in `<repo>/.env` and run with:

| model | settings |
|---|---|
| GPT-5.6-Luna | `REMOTE_MODEL=openai/gpt-5.6-luna REMOTE_REASONING=none REMOTE_TEMPERATURE=none` |
| Claude-Sonnet-5 | `REMOTE_MODEL=anthropic/claude-sonnet-5 REMOTE_REASONING=none REMOTE_TEMPERATURE=none` |
| Kimi-K2.6 | `REMOTE_MODEL=moonshotai/kimi-k2.6 REMOTE_REASONING=none REMOTE_TEMPERATURE=0.4` |

## Part 1: prompting conditions (Sec. 4.3 (1)-(3), Table 1)

Test split, both languages, all 13 prompts. Run for all eight models and both conflicts.

```bash
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/run_baseline.sh
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/run_neutrality_control.sh
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/run_self_distancing.sh
PAIR=ru_ua MODEL_NAME=gpt-5.6-luna REMOTE_MODEL=openai/gpt-5.6-luna REMOTE_REASONING=none REMOTE_TEMPERATURE=none \
  bash scripts/run_baseline.sh
```

## Part 2: SDAS (Sec. 3.2, 4.3 (4), App. C)

**1. Direction, train split (App. C.1).** Paired generation (original vs SD1-prepended prompt, same seed) and the
per-layer mean activation difference (Eq. 1-2).

```bash
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/run_sdas_train.sh
```

**2. Layer and strength, validation split (App. C.2-C.3).** The first call prints the teacher-forced layer screen
(preference gain, preservation cost); choose candidate layers from it. The second call runs the grid and prints the
cell with the smallest |g_val| (Eq. 14).

```bash
PAIR=ru_ua MODEL_NAME=qwen25-7b bash scripts/run_sdas_validation.sh
PAIR=ru_ua MODEL_NAME=qwen25-7b LAYERS=11,12,16,17 bash scripts/run_sdas_validation.sh
```

Paper layers and cells (Tables 8-11; zero-based layers, `L17_a1` = layer 17, alpha 1):

| model | RU-UA `LAYERS` | RU-UA cell | IL-PS `LAYERS` | IL-PS cell |
|---|---|---|---|---|
| qwen25-7b | `11,12,16,17` | `L17_a1` | `12,16,17` | `L16_a1` |
| falcon3-7b-instruct | `7,14,16` | `L16_a2` | `8,14,16` | `L16_a2` |
| mistral-7b-instruct-v0.3 | `8,16,19` | `L16_a2` | `8,16,18` | `L16_a2` |
| aya-expanse-8b | `14,16,18` | `L14_a2` | `14,15,18` | `L14_a2` |
| gemma-4-e4b-it | `13,18,20,22` | `L13_a2` | `12,13,18,20` | `L12_a2` |

For Gemma-4, choose layers below 24 (its last 18 layers reuse earlier KV states).

**3. Test split (Sec. 5.1).** Needs the baseline for the same model.

```bash
PAIR=ru_ua MODEL_NAME=qwen25-7b CELL=L17_a1 bash scripts/run_sdas_test.sh
```

Steering adds alpha * d to the output of layer l at the last prompt token and every generated token (Eq. 3,
`SteerHook` in `src/sdas/hooked_model.py`).

## Part 3: cross-conflict alignment and transfer (Sec. 5.2, Tables 2-3)

Needs SDAS step 1 on both conflicts.

```bash
MODELS="qwen25-7b:17,16" bash scripts/run_sdas_alignment.sh
TRANSFER=ru_ua_to_il_ps MODEL_NAME=qwen25-7b CELL=L17_a1 bash scripts/run_sdas_transfer.sh
TRANSFER=il_ps_to_ru_ua MODEL_NAME=qwen25-7b CELL=L16_a1 bash scripts/run_sdas_transfer.sh
```

Transfer applies the source direction at the source cell to the target test split; it needs `run_baseline.sh` on the
target conflict.

## Part 4: ablations and robustness

```bash
VARIANT=v2 PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=... bash scripts/run_self_distancing.sh   # SD2 (App. I)
VARIANT=v3 PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=... bash scripts/run_self_distancing.sh   # SD3
PAIR=ru_ua JUDGE_PROMPT=EVENT_DISTANCE_JUDGE bash scripts/run_judge_robustness.sh               # judge prompt (App. D.6)
PAIR=ru_ua JUDGE_NAME=falcon3-7b-instruct JUDGE_PATH=/path/to/Falcon3-7B-Instruct bash scripts/run_judge_robustness.sh
PAIR=ru_ua JUDGE_NAME=gpt-5.1 JUDGE_PATH= REMOTE_JUDGE=openai/gpt-5.1 bash scripts/run_judge_robustness.sh
```

Judge robustness re-scores stored outputs of every condition and model; run it once per conflict after all models.

## Annotation (Sec. 4.5-4.6, App. E, G)

```bash
PAIR=ru_ua TASK=preservation bash scripts/run_annotation.sh   # Tables 16-17
PAIR=ru_ua TASK=mechanisms bash scripts/run_annotation.sh     # Tables 4, 20-22
```

Prompts: `src/annotation/prompts.py`.

## Statistics (Sec. 4.4, 4.7, App. D, H)

Scores are distances d in [0, 1] (the paper reports s = 1 - d; |g| is identical). Effect: |g_intervention| -
|g_baseline| (Eq. 5). 95% CI: event-clustered bootstrap, 2,000 resamples. p: event-level paired permutation, 5,000
permutations. Benjamini-Hochberg within each conflict.

`src/pipeline/sdas/test_stats.py` (called by `run_sdas_test.sh`, `run_sdas_transfer.sh`). Paper statistic: `d_absgap`,
`ci_lo`, `ci_hi`, `p`, `q`. Neutrality control:

```bash
python src/pipeline/sdas/test_stats.py --pair ru_ua --injection control --include_injection
```

Notebooks (`src/statistics/`, set `PAIR` in the first cell):

| paper | notebook | notes |
|---|---|---|
| Table 1, Table 23 | `calculate_gap_test_sdas.ipynb` | paper statistic: `abs_gap_stats()`; neutrality: `INJECTION_MITIGATION = "control"` |
| Table 24 | `calculate_gap_test_sdas_vs_self_distance.ipynb` | paper statistic: `abs_gap_stats()` |
| Table 25 | `self_distance_ablations/calculate_gap_self_distance_ablation.ipynb` | |
| App. D.6-D.7 | `calculate_gap_test_sdas.ipynb` with `JUDGE_MODEL` / `JUDGE_PROMPT`; `calculate_kappa.ipynb` | |
| Tables 2, 3, 8-11 | `run_sdas_alignment.sh`, `run_sdas_transfer.sh`, `run_sdas_validation.sh` | |

## Released SDAS vectors

`SDAS_models/<model>/layer_<LL>.pt`: the RU-UA direction at the selected layer (alpha 1 for Qwen, 2 for the others).

```bash
python src/pipeline/sdas/sdas_generate.py --model_name qwen25-7b --model_path /path/to/Qwen2.5-7B-Instruct \
    --text "Russian strikes hit energy facilities in Kharkiv." --lang ua --context twitter_1 --compare
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/run_sdas_released.sh
```
