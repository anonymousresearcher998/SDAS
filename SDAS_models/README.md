# SDAS_models

Steering vectors learned on the RU/UA conflict. These are the exact vectors used in the original RU/UA
steering runs and, unchanged, in the cross-conflict transfer runs.

Environments: `environment.yml` (repository root) for every model here except `gemma-4-e4b-it`, which needs
`environment-gemma.yml` (transformers 5.x) - see "Environments" in the root `README.md`.

## Layout

```
SDAS_models/
  <model>/
    layer_LL.pt
```

One file per model: a bare 1-D `float32` torch tensor of length `hidden_size`, the vector for decoder
layer `LL` (0-based). Load with `torch.load(path, weights_only=True)`.

## Contents

| model directory             | file          | vector length | alpha used in RU/UA |
|-----------------------------|---------------|---------------|---------------------|
| `aya-expanse-8b`            | `layer_14.pt` | 4096          | 2                   |
| `falcon3-7b-instruct`       | `layer_16.pt` | 3072          | 2                   |
| `gemma-4-e4b-it`            | `layer_13.pt` | 2560          | 2                   |
| `mistral-7b-instruct-v0.3`  | `layer_16.pt` | 4096          | 2                   |
| `qwen25-7b`                 | `layer_17.pt` | 3584          | 1                   |

The layer and alpha are the single cell selected on RU/UA and reported for that model. The same file
and alpha were used for the cross-conflict transfer.

## How the vector is applied

At the stored layer `l`, with the vector `v` and scale `alpha`:

```
h_l  <-  h_l + alpha * v        # output hidden state of decoder layer l
```

applied at the response positions (the last prompt token and every generated token) during decoding.
`alpha = 0` is an exact no-op. The vectors are stored as learned, not unit-normalized, so `alpha`
multiplies the raw vector.

## Using the vectors

**Generate with one prompt** - loads the model and its vector here, and steers its one layer with its alpha
(override with `--vector` / `--alpha`); `--compare` also prints the unsteered output with the same seed:

```bash
# the paper's task: an English event statement, a target language (ru | ua | ps | il) and a communication context
python src/pipeline/sdas/sdas_generate.py --model_name qwen25-7b --model_path /path/to/Qwen2.5-7B-Instruct \
    --text "Russian strikes hit energy facilities in Kharkiv." --lang ua --context twitter_1 --compare
# any prompt, through the model's chat template
python src/pipeline/sdas/sdas_generate.py --model_name qwen25-7b --model_path /path/to/Qwen2.5-7B-Instruct --prompt "..."
```

**The held-out test evaluation** (paper Sec. 5.1) with these vectors, without training or validation - see
`scripts/run_sdas_released.sh`:

```bash
PAIR=ru_ua MODEL_NAME=qwen25-7b MODEL_PATH=/path/to/Qwen2.5-7B-Instruct JUDGE_PATH=/path/to/Llama-3.1-8B-Instruct \
  bash scripts/run_sdas_released.sh
```

With `PAIR=il_ps` this is the RU/UA -> IL/PS transfer. For `gemma-4-e4b-it`, use the `environment-gemma.yml`
environment for generation (`PYTHON`, and for `sdas_generate.py` its interpreter).
