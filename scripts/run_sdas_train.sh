source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME=<experiment name>, e.g. qwen25-7b}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=<local weights>}"
INTERVENTION="${INTERVENTION:-self_distance_v1}"
SEED="${SEED:-1234}"
LAYERS="${LAYERS:-all}"
EXP_DIR="$ROOT/sdas/$MODEL_NAME"

for lang in "$LANG_A" "$LANG_B"; do
  args=( --data "$DATA_PATTERN" --sets train,val,test --local_translator_model "$MODEL_PATH"
         --translator_model_name "$MODEL_NAME" --lang "$lang" --prompts all --layers "$LAYERS"
         --intervention "$INTERVENTION" --seed "$SEED" --dtype pipeline --out_root "$ROOT/sdas" )
  [[ -n "$LIMIT" ]] && args+=( --limit "$LIMIT" )
  run "$PYTHON" src/pipeline/sdas/generate_pairs.py "${args[@]}"
done

run "$PYTHON" src/pipeline/sdas/learn_direction.py --experiment_dir "$EXP_DIR" --name "$INTERVENTION" \
  --intervention "$INTERVENTION" --data "$DATA_PATTERN" --train train --overwrite

echo "direction: $EXP_DIR/directions/$INTERVENTION/direction.pt"
