source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<model name, as in SDAS_models/>, e.g. qwen25-7b}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=<local weights>}"
SEED="${SEED:-1234}"
DIRECTION="sdas_models"

if [[ -z "${DIRECTION_FILE:-}" ]]; then
  shopt -s nullglob; files=( "SDAS_models/$MODEL_NAME"/layer_*.pt ); shopt -u nullglob
  [[ ${#files[@]} -eq 1 ]] || fatal "expected one SDAS_models/$MODEL_NAME/layer_*.pt (found ${#files[@]}) - set DIRECTION_FILE"
  DIRECTION_FILE="${files[0]}"
fi
[[ -f "$DIRECTION_FILE" ]] || fatal "released vector not found: $DIRECTION_FILE"
[[ "$(basename "$DIRECTION_FILE")" =~ ^layer_([0-9]+)\.pt$ ]] || fatal "a released vector is named layer_<LL>.pt (got $DIRECTION_FILE)"
LAYER=$(( 10#${BASH_REMATCH[1]} ))
if [[ -z "${ALPHA:-}" ]]; then
  case "$MODEL_NAME" in
    qwen25-7b) ALPHA=1 ;;
    aya-expanse-8b|falcon3-7b-instruct|gemma-4-e4b-it|mistral-7b-instruct-v0.3) ALPHA=2 ;;
    *) fatal "no released alpha for $MODEL_NAME - set ALPHA" ;;
  esac
fi
EXP_DIR="$ROOT/sdas/$MODEL_NAME"
SLUG="$(printf 'sdas/%s/L%02d_a%s' "$DIRECTION" "$LAYER" "$(awk "BEGIN { printf \"%g\", $ALPHA }")")"
require_judge
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=( --limit "$LIMIT" )
echo "released vector: $DIRECTION_FILE (layer $LAYER), alpha $ALPHA -> $SLUG on the $PAIR test split"
[[ "$PAIR" == "il_ps" ]] && echo "note: the released vectors were learned on RU-UA - on IL-PS this is the RU-UA -> IL-PS transfer (Table 3)"

for lang in "$LANG_A" "$LANG_B"; do
  # (a) unsteered test generation (the seeds of the steered rows) - GPU; existing rows are kept
  run "$PYTHON" src/pipeline/sdas/generate_pairs.py --data "$DATA_PATTERN" --sets test --regular_only \
    --local_translator_model "$MODEL_PATH" --translator_model_name "$MODEL_NAME" --lang "$lang" --prompts all \
    --layers all --seed "$SEED" --dtype pipeline --out_root "$ROOT/sdas" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  # (b) steered test generation with the released vector - GPU
  run "$PYTHON" src/pipeline/sdas/generate_sdas.py --experiment_dir "$EXP_DIR" --direction "$DIRECTION" \
    --direction_file "$DIRECTION_FILE" --layers "$LAYER" "--alphas=$ALPHA" --positions response --lang "$lang" \
    --prompts all --ids test --data "$DATA_PATTERN" --local_translator_model "$MODEL_PATH" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  # (c) judge + export - GPU
  run "$JUDGE_PYTHON" src/pipeline/sdas/judge.py --experiment_dir "$EXP_DIR" --condition "$SLUG" --lang "$lang" --ids test \
    --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" --data "$DATA_PATTERN" \
    --local_judge_model "$JUDGE_PATH" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
done

# (d) statistics vs. the baseline
run "$JUDGE_PYTHON" src/pipeline/sdas/test_stats.py --pair "$PAIR" --models "$MODEL_NAME" --direction "$DIRECTION" \
  --judge "$JUDGE_NAME" --judge_prompt "$JUDGE_PROMPT"
