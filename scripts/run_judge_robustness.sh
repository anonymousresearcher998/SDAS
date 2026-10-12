source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REMOTE_JUDGE="${REMOTE_JUDGE:-}"
TREES="${TREES:-baseline,control,self_distance,sdas_test}"
MODELS="${MODELS:-}"
SOURCE_JUDGE_NAME="${SOURCE_JUDGE_NAME:-llama3-8b}"
SOURCE_JUDGE_PROMPT="${SOURCE_JUDGE_PROMPT:-BASE_JUDGE}"

if [[ -n "$REMOTE_JUDGE" ]]; then
  [[ -z "$JUDGE_PATH" ]] || fatal "set JUDGE_PATH (local judge) or REMOTE_JUDGE (API judge), not both"
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || fatal "REMOTE_JUDGE needs OPENROUTER_API_KEY (environment or .env)"
  JUDGE_ARGS=( --remote_judge_model "$REMOTE_JUDGE" )
else
  require_judge
  JUDGE_ARGS=( --local_judge_model "$JUDGE_PATH" )
fi
[[ "$JUDGE_NAME" != "$SOURCE_JUDGE_NAME" || "$JUDGE_PROMPT" != "$SOURCE_JUDGE_PROMPT" ]] \
  || fatal "($JUDGE_NAME / $JUDGE_PROMPT) is the primary one"
[[ -n "${JUDGE_DTYPE:-}" ]] && JUDGE_ARGS+=( --judge_dtype "$JUDGE_DTYPE" )
[[ -n "$LIMIT" ]] && JUDGE_ARGS+=( --limit "$LIMIT" )

IFS=',' read -ra TREE_ARR <<< "$TREES"
for tree in "${TREE_ARR[@]}"; do
  source_dir="$ROOT/$tree/${SOURCE_JUDGE_NAME}_judge/$SOURCE_JUDGE_PROMPT"
  if [[ ! -d "$source_dir" ]]; then echo "skip $tree: no primary scores under $source_dir"; continue; fi
  if [[ -n "$MODELS" ]]; then
    IFS=',' read -ra MODEL_ARR <<< "$MODELS"
  else
    MODEL_ARR=(); for d in "$source_dir"/*/; do MODEL_ARR+=( "$(basename "$d")" ); done
  fi
  for model in "${MODEL_ARR[@]}"; do
    [[ -d "$source_dir/$model" ]] || { echo "skip $tree/$model: not found"; continue; }
    for lang in "$LANG_A" "$LANG_B"; do
      run "$JUDGE_PYTHON" src/pipeline/en_judge_same.py --lang "$lang" --data "$(data_file test)" \
        --precomputed_mt_root "$source_dir/$model" --translator_model_name "$model" --mitigation "$tree" \
        --out_dir "$ROOT" --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" "${JUDGE_ARGS[@]}"
    done
  done
done
