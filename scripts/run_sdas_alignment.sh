source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODELS="${MODELS:-qwen25-7b:17,16 falcon3-7b-instruct:16,16 mistral-7b-instruct-v0.3:16,16 aya-expanse-8b:14,14 gemma-4-e4b-it:13,12}"
INTERVENTION="${INTERVENTION:-self_distance_v1}"
RU_UA_ROOT="${RU_UA_ROOT:-results/sdas}"
IL_PS_ROOT="${IL_PS_ROOT:-results_il_ps/sdas}"

MODEL_ARGS=()
for m in $MODELS; do MODEL_ARGS+=( --model "$m" ); done
[[ ${#MODEL_ARGS[@]} -gt 0 ]] || fatal "set MODELS, e.g. MODELS=\"qwen25-7b:17,16\""

run "$JUDGE_PYTHON" src/pipeline/sdas/cross_conflict_alignment.py "${MODEL_ARGS[@]}" --direction "$INTERVENTION" \
  --ru_ua_root "$RU_UA_ROOT" --il_ps_root "$IL_PS_ROOT"
