source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASK="${TASK:-}"
ANNOTATOR="${ANNOTATOR:-openai/gpt-5.6-luna}"
TREES="${TREES:-baseline,control,self_distance,sdas_test}"
MODELS="${MODELS:-}"

[[ "$TASK" == "mechanisms" || "$TASK" == "preservation" ]] || fatal "set TASK=mechanisms or TASK=preservation"
[[ -n "${OPENROUTER_API_KEY:-}" || "$DRY_RUN" == "1" ]] || fatal "needs OPENROUTER_API_KEY (environment or .env)"

ARGS=( --task "$TASK" --root "$ROOT" --langs "$LANG_A,$LANG_B" --annotator "$ANNOTATOR" --trees "$TREES"
       --judge_name "$JUDGE_NAME" --judge_prompt "$JUDGE_PROMPT" )
[[ -n "$MODELS" ]] && ARGS+=( --models "$MODELS" )
[[ -n "$LIMIT" ]] && ARGS+=( --limit "$LIMIT" )

run "$JUDGE_PYTHON" src/pipeline/annotate.py "${ARGS[@]}"
