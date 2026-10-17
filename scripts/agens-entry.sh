#!/bin/bash
# Container entry point: --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [-- extra agens serve flags]
# Environment variables already set win over the profile's (AGENS_* tuning knobs).
set -euo pipefail
PROFILE=16gb; MODEL=/model; DRAFTER=; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done
setdef() { [ -n "${!1:-}" ] || export "$1=$2"; }
case "$PROFILE" in
  16gb) ARGS=(--budget-gib 16 --wctx 262148 --mode personal --no-fork 1); setdef AGENS_KV_INT8_DENSE 1;;
  12gb) ARGS=(--budget-gib 12 --wctx "${AGENS_WCTX:-65540}" --mode personal --no-fork 1); setdef AGENS_KV_INT8_DENSE 1
        [ "${AGENS_12GB_DRAFTER:-0}" = 1 ] || DRAFTER=;;
  24gb) ARGS=(--budget-gib 23 --wctx 262148 --mode server --max-batch 4 --max-convs 4);;
  48gb) ARGS=(--budget-gib 46 --wctx 262148 --mode server --max-batch 16 --max-convs 16);;
  *) echo "unknown profile $PROFILE (12gb, 16gb, 24gb, 48gb)" >&2; exit 2;;
esac
[ -n "$DRAFTER" ] && ARGS+=(--spec-drafter "$DRAFTER")
exec agens serve "$MODEL" --host 0.0.0.0 --port 30000 "${ARGS[@]}" "${EXTRA[@]}"
