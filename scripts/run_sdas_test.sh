source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<experiment name>, e.g. qwen25-7b}"
CELL="${CELL:?set CELL=L<layer>_a<alpha>, e.g. CELL=L17_a1 (from run_sdas_validation.sh)}"
INTERVENTION="${INTERVENTION:-self_distance_v1}"
[[ "$CELL" =~ ^L([0-9]+)_a(-?[0-9]+(\.[0-9]+)?)$ ]] || fatal "CELL must look like L17_a1 (got '$CELL')"
LAYER=$(( 10#${BASH_REMATCH[1]} )); ALPHA="${BASH_REMATCH[2]}"
EXP_DIR="$ROOT/sdas/$MODEL_NAME"
SLUG="$(printf 'sdas/%s/L%02d_a%s' "$INTERVENTION" "$LAYER" "$(awk "BEGIN { printf \"%g\", $ALPHA }")")"
require_judge
[[ "$DRY_RUN" == "1" || -f "$EXP_DIR/directions/$INTERVENTION/direction.pt" ]] || fatal "no direction in $EXP_DIR - run scripts/run_sdas_train.sh first"
MODEL_ARGS=(); [[ -n "${MODEL_PATH:-}" ]] && MODEL_ARGS=( --local_translator_model "$MODEL_PATH" )
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=( --limit "$LIMIT" )

for lang in "$LANG_A" "$LANG_B"; do
  # (a) steered generation on test - GPU
  run "$PYTHON" src/pipeline/sdas/generate_sdas.py --experiment_dir "$EXP_DIR" --direction "$INTERVENTION" \
    --layers "$LAYER" "--alphas=$ALPHA" --positions response --lang "$lang" --prompts all --ids test \
    --data "$DATA_PATTERN" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  # (b) judge + export - GPU
  run "$JUDGE_PYTHON" src/pipeline/sdas/judge.py --experiment_dir "$EXP_DIR" --condition "$SLUG" --lang "$lang" --ids test \
    --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" --data "$DATA_PATTERN" \
    --local_judge_model "$JUDGE_PATH" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
done

# (c) statistics
run "$JUDGE_PYTHON" src/pipeline/sdas/test_stats.py --pair "$PAIR" --models "$MODEL_NAME" \
  --judge "$JUDGE_NAME" --judge_prompt "$JUDGE_PROMPT"
