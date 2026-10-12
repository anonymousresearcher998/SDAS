set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python}"
JUDGE_PYTHON="${JUDGE_PYTHON:-$PYTHON}"
PAIR="${PAIR:-ru_ua}"
JUDGE_PATH="${JUDGE_PATH:-}"
JUDGE_NAME="${JUDGE_NAME:-llama3-8b}"
JUDGE_PROMPT="${JUDGE_PROMPT:-BASE_JUDGE}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

fatal() { echo "FATAL: $*" >&2; exit 1; }

set_pair() {
  case "$1" in
    ru_ua) ROOT="results";       DATA_PATTERN="data/ru_ua/neutral_data_{split}.json"; LANG_A=ru; LANG_B=ua ;;
    il_ps) ROOT="results_il_ps"; DATA_PATTERN="data/il_ps/neutral_data_{split}.json"; LANG_A=ps; LANG_B=il ;;
    *) fatal "PAIR must be ru_ua or il_ps (got '$1')" ;;
  esac
}
set_pair "$PAIR"

data_file() { echo "${DATA_PATTERN//\{split\}/$1}"; }

require_judge() {
  [[ -n "$JUDGE_PATH" ]] || fatal "set JUDGE_PATH=<local weights of the judge model>"
  [[ "$DRY_RUN" == "1" || -d "$JUDGE_PATH" ]] || fatal "judge weights not found: $JUDGE_PATH"
}

run() {
  printf '+'; printf ' %q' "$@"; echo
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

run_prompting() {
  local mitigation="$1" out_dir="$2" lang args
  local model_name="${MODEL_NAME:-}" model_path="${MODEL_PATH:-}" remote_model="${REMOTE_MODEL:-}"
  [[ -n "$model_name" ]] || fatal "set MODEL_NAME=<results bucket name>, e.g. MODEL_NAME=qwen25-7b"
  [[ -n "$model_path" && -z "$remote_model" || -z "$model_path" && -n "$remote_model" ]] \
    || fatal "set exactly one of MODEL_PATH (local weights) or REMOTE_MODEL (OpenRouter slug)"
  [[ -z "$remote_model" || -n "${OPENROUTER_API_KEY:-}" || "$DRY_RUN" == "1" ]] \
    || fatal "REMOTE_MODEL needs OPENROUTER_API_KEY (environment or .env)"
  require_judge
  for lang in "$LANG_A" "$LANG_B"; do
    args=( --data "$(data_file test)" --translator_model_name "$model_name" --out_dir "$out_dir"
           --mitigation "$mitigation" --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT"
           --local_judge_model "$JUDGE_PATH" )
    if [[ -n "$model_path" ]]; then
      args+=( --local_translator_model "$model_path" )
    else
      args+=( --remote_translator_model "$remote_model" )
      [[ -n "${REMOTE_REASONING:-}" ]] && args+=( --remote_reasoning "$REMOTE_REASONING" )
      [[ -n "${REMOTE_TEMPERATURE:-}" ]] && args+=( --remote_temperature "$REMOTE_TEMPERATURE" )
    fi
    [[ -n "${TRANSLATOR_DTYPE:-}" ]] && args+=( --translator_dtype "$TRANSLATOR_DTYPE" )
    [[ -n "${JUDGE_DTYPE:-}" ]] && args+=( --judge_dtype "$JUDGE_DTYPE" )
    args+=( --seed "${SEED:-1234}" )   # base seed of the per-item generation seeds, as in run_sdas_train.sh
    [[ -n "$LIMIT" ]] && args+=( --limit "$LIMIT" )
    run "$PYTHON" "src/pipeline/en_to_${lang}_bias.py" "${args[@]}"
  done
  echo "results: $out_dir/$mitigation/${JUDGE_NAME}_judge/$JUDGE_PROMPT/$model_name/<prompt>/<lang>_run_<timestamp>/"
}

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export TOKENIZERS_PARALLELISM=false
