source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRANSFER="${TRANSFER:?set TRANSFER=ru_ua_to_il_ps or TRANSFER=il_ps_to_ru_ua}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<experiment name>, e.g. qwen25-7b}"
CELL="${CELL:?set CELL=L<layer>_a<alpha>, the cell selected on the source pair}"
INTERVENTION="${INTERVENTION:-self_distance_v1}"
case "$TRANSFER" in
  ru_ua_to_il_ps) SRC_PAIR=ru_ua; DST_PAIR=il_ps ;;
  il_ps_to_ru_ua) SRC_PAIR=il_ps; DST_PAIR=ru_ua ;;
  *) fatal "TRANSFER must be ru_ua_to_il_ps or il_ps_to_ru_ua (got '$TRANSFER')" ;;
esac
[[ "$CELL" =~ ^L([0-9]+)_a(-?[0-9]+(\.[0-9]+)?)$ ]] || fatal "CELL must look like L17_a1 (got '$CELL')"
LAYER=$(( 10#${BASH_REMATCH[1]} )); ALPHA="${BASH_REMATCH[2]}"

set_pair "$SRC_PAIR"; SRC_ROOT="$ROOT"
set_pair "$DST_PAIR"
DIRECTION_DIR="$SRC_ROOT/sdas/$MODEL_NAME/directions/$INTERVENTION"
DIRECTION="${INTERVENTION}_${SRC_PAIR}"
DST_MAIN="$ROOT/sdas/$MODEL_NAME"
DST_EXP="$ROOT/cross_conflict/sdas/$MODEL_NAME"
SLUG="$(printf 'sdas/%s/L%02d_a%s' "$DIRECTION" "$LAYER" "$(awk "BEGIN { printf \"%g\", $ALPHA }")")"
require_judge
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$DIRECTION_DIR/direction.pt" ]] || fatal "source direction not found: $DIRECTION_DIR (run run_sdas_train.sh with PAIR=$SRC_PAIR)"
  [[ -f "$DST_MAIN/experiment.json" ]] || fatal "target experiment not found: $DST_MAIN (run run_sdas_train.sh with PAIR=$DST_PAIR)"
fi
MODEL_ARGS=(); [[ -n "${MODEL_PATH:-}" ]] && MODEL_ARGS=( --local_translator_model "$MODEL_PATH" )
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=( --limit "$LIMIT" )

run "$PYTHON" src/pipeline/sdas/seed_from_experiment.py --src_experiment_dir "$DST_MAIN" --dst_experiment_dir "$DST_EXP" \
  --data "$DATA_PATTERN"

for lang in "$LANG_A" "$LANG_B"; do
  run "$PYTHON" src/pipeline/sdas/generate_sdas.py --experiment_dir "$DST_EXP" --direction "$DIRECTION" \
    --direction_dir "$DIRECTION_DIR" --layers "$LAYER" "--alphas=$ALPHA" --positions response --lang "$lang" \
    --prompts all --ids test --data "$DATA_PATTERN" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  run "$JUDGE_PYTHON" src/pipeline/sdas/judge.py --experiment_dir "$DST_EXP" --condition "$SLUG" --lang "$lang" --ids test \
    --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" --data "$DATA_PATTERN" \
    --local_judge_model "$JUDGE_PATH" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
done

run "$JUDGE_PYTHON" src/pipeline/sdas/test_stats.py --pair "$DST_PAIR" --models "$MODEL_NAME" \
  --sdas cross_conflict/sdas_test --direction "$DIRECTION" --judge "$JUDGE_NAME" --judge_prompt "$JUDGE_PROMPT"
