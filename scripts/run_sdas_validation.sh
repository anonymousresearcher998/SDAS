source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<experiment name>, e.g. qwen25-7b}"
INTERVENTION="${INTERVENTION:-self_distance_v1}"
SCREEN_N_PAIRS="${SCREEN_N_PAIRS:-40}"
LAYERS="${LAYERS:-}"
ALPHAS="${ALPHAS:-1,2,4}"
EXP_DIR="$ROOT/sdas/$MODEL_NAME"
require_judge
[[ "$DRY_RUN" == "1" || -f "$EXP_DIR/directions/$INTERVENTION/direction.pt" ]] || fatal "no direction in $EXP_DIR - run scripts/run_sdas_train.sh first"
MODEL_ARGS=(); [[ -n "${MODEL_PATH:-}" ]] && MODEL_ARGS=( --local_translator_model "$MODEL_PATH" )
LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=( --limit "$LIMIT" )
slug() { printf 'sdas/%s/L%02d_a%s' "$INTERVENTION" "$1" "$(awk "BEGIN { printf \"%g\", $2 }")"; }

if [[ -z "$LAYERS" ]]; then
  run "$PYTHON" src/pipeline/sdas/screen_layers.py --experiment_dir "$EXP_DIR" --direction "$INTERVENTION" \
    --n_pairs "$SCREEN_N_PAIRS" --alphas=1,2,4 --ids val --positions response --prompts all ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
  echo
  echo "Screen done ($EXP_DIR/screens/$INTERVENTION/). Choose the candidate layers from the table above, then run"
  echo "  LAYERS=<l1,l2,...> MODEL_NAME=$MODEL_NAME PAIR=$PAIR bash scripts/run_sdas_validation.sh"
  exit 0
fi

[[ "$LAYERS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fatal "LAYERS must be comma-separated layer indices, e.g. LAYERS=11,12,16,17 (got '$LAYERS')"
IFS=',' read -ra L_ARR <<< "$LAYERS"
IFS=',' read -ra A_ARR <<< "$ALPHAS"
NOISE_LAYER="${L_ARR[1]:-${L_ARR[0]}}"
echo "candidate layers: ${L_ARR[*]}   alphas: ${A_ARR[*]}   unsteered reference: layer $NOISE_LAYER, alpha 0, replicate 1"

CONDITIONS=()
for lang in "$LANG_A" "$LANG_B"; do
  for layer in "${L_ARR[@]}"; do
    for alpha in "${A_ARR[@]}"; do
      run "$PYTHON" src/pipeline/sdas/generate_sdas.py --experiment_dir "$EXP_DIR" --direction "$INTERVENTION" \
        --layers "$layer" "--alphas=$alpha" --positions response --lang "$lang" --prompts all --ids val \
        --data "$DATA_PATTERN" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
    done
  done
  run "$PYTHON" src/pipeline/sdas/generate_sdas.py --experiment_dir "$EXP_DIR" --direction "$INTERVENTION" \
    --layers "$NOISE_LAYER" --alphas=0 --rep 1 --positions response --lang "$lang" --prompts all --ids val \
    --data "$DATA_PATTERN" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
done
for layer in "${L_ARR[@]}"; do
  for alpha in "${A_ARR[@]}"; do CONDITIONS+=( --condition "$(slug "$layer" "$alpha")" ); done
done
REFERENCE="$(slug "$NOISE_LAYER" 0)_r1"
CONDITIONS+=( --condition "$REFERENCE" )

for lang in "$LANG_A" "$LANG_B"; do
  run "$JUDGE_PYTHON" src/pipeline/sdas/judge.py --experiment_dir "$EXP_DIR" "${CONDITIONS[@]}" --lang "$lang" --ids val \
    --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" --data "$DATA_PATTERN" \
    --local_judge_model "$JUDGE_PATH" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
done
SUMMARY_DIR="$EXP_DIR/summary/val_$(date +%Y-%m-%d_%H-%M-%S)"
run "$JUDGE_PYTHON" src/pipeline/sdas/summarize.py --experiment_dir "$EXP_DIR" --langs "$LANG_A,$LANG_B" --ids val \
  --reference "$REFERENCE" --judge_model "$JUDGE_NAME" --judge_prompt_name "$JUDGE_PROMPT" --out "$SUMMARY_DIR"

[[ "$DRY_RUN" == "1" ]] && exit 0
"$PYTHON" - "$SUMMARY_DIR/summary_gap.csv" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))
        if r["kind"] == "sdas" and r["gap"] and r["alpha"] and float(r["alpha"]) != 0 and not r["rep"]]
if not rows:
    sys.exit("no judged steering cell in " + sys.argv[1])
print("\nvalidation |gap| per cell:")
for r in sorted(rows, key=lambda r: abs(float(r["gap"]))):
    print(f"  {r['condition']:40s} |gap| = {abs(float(r['gap'])):.4f}")
best = min(rows, key=lambda r: abs(float(r["gap"])))
cell = best["condition"].rsplit("/", 1)[1]
print(f"\nselected cell: {cell}   ->   CELL={cell} bash scripts/run_sdas_test.sh")
EOF
