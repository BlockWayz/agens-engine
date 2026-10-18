#!/bin/bash
# Container entry point: --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [--images on|off]
#                        [-- extra agens serve flags]
# --images: image input (image_url parts; README, Image input). Default: on for 24gb / 48gb, off for 12gb / 16gb. On a 16gb
# card it costs KV pool for the image-encode workspace: the context drops to ~225K (AGENS_IMAGE_MAX_PIXELS 4.19 MP).
# Environment variables already set win over the profile's (AGENS_* tuning knobs).
set -euo pipefail
PROFILE=16gb; MODEL=/model; DRAFTER=; IMAGES=; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --images) IMAGES=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done
setdef() { [ -n "${!1:-}" ] || export "$1=$2"; }
case "$PROFILE" in
  16gb) # image input on: the encode workspace (0.37 GB at 4.19 MP) comes out of the KV pool -> context ~225K
        if [ "${IMAGES:-off}" = on ]; then W16=${AGENS_WCTX:-229380}; else W16=${AGENS_WCTX:-262148}; fi
        ARGS=(--budget-gib 16 --wctx "$W16" --mode personal --no-fork 1); setdef AGENS_KV_INT8_DENSE 1;;
  12gb) ARGS=(--budget-gib 12 --wctx "${AGENS_WCTX:-65540}" --mode personal --no-fork 1 --spec-copy 1)
        setdef AGENS_KV_INT8_DENSE 1; setdef AGENS_SPEC_FAR_CACHE 0  # no far-branch union cache (saves ~140 MB)
        [ "${AGENS_12GB_DRAFTER:-0}" = 1 ] || DRAFTER=;;  # copy-from-context drafting only: no drafter weights
  # 24gb: 4 KDA state slots (serve's default is 2 x --max-batch = 8, which with the drafter leaves no KV pool at
  # 23 GiB for the 4-bit release weights: 1.0.0 failed to start "budget 23 GiB leaves no room for a KV pool")
  24gb) ARGS=(--budget-gib 23 --wctx 262148 --mode server --max-batch 4 --max-convs 4 --sessions 4);;
  48gb) ARGS=(--budget-gib 46 --wctx 262148 --mode server --max-batch 16 --max-convs 16);;
  *) echo "unknown profile $PROFILE (12gb, 16gb, 24gb, 48gb)" >&2; exit 2;;
esac
case "${IMAGES:-}" in
  on) ARGS+=(--vision 1);;
  off) ARGS+=(--vision 0);;
  "") case "$PROFILE" in 12gb|16gb) ARGS+=(--vision 0);; esac;;  # 24gb / 48gb: on when the checkpoint has the tower
  *) echo "--images on|off (got $IMAGES)" >&2; exit 2;;
esac
[ -n "$DRAFTER" ] && ARGS+=(--spec-drafter "$DRAFTER")
exec agens serve "$MODEL" --host 0.0.0.0 --port 30000 "${ARGS[@]}" "${EXTRA[@]}"
