source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VARIANT="${VARIANT:-v1}"
case "$VARIANT" in
  v1)    run_prompting self_distance "$ROOT" ;;
  v2|v3) run_prompting "self_distance_$VARIANT" "$ROOT/self_distance_ablation" ;;
  *)     fatal "VARIANT must be v1, v2 or v3 (got '$VARIANT')" ;;
esac
