#!/bin/bash
# Container entry point: --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [--mtp DIR] [--images on|off]
#                        [--tp 1|2|4] [-- extra agens serve flags]
# --mtp: the model's MTP head (docs/mtp.md; mount it like the drafter, e.g. -v /path/mtp:/mtp:ro --mtp /mtp). It joins the
# drafting chain (context copy, MTP, DFlash2 drafter) in the 16gb, 24gb and 48gb profiles; the 12gb profile ignores it.
# --tp N: tensor parallel over the container's first N GPUs (one process per GPU; `docker run --gpus '"device=0,1"'
# --ipc=host`). The profile's budget is per GPU. Under --tp the KV cache stays on the GPUs (each holds 1 / N of the K/V
# heads; no host tiering), so the per-GPU plan has more context or more conversation slots than tp1 on the same card.
# --mtp with --tp 2|4: not supported in 1.2; the entry point drops --mtp (with a message) and runs DFlash2 drafting under --tp.
# --images: image input (image_url parts; README, Image input). Default: on for 24gb / 48gb, off for 12gb / 16gb. On a 16gb
# card it costs KV pool for the image-encode workspace: the context drops to ~225K (AGENS_IMAGE_MAX_PIXELS 4.19 MP).
# Environment variables already set win over the profile's (AGENS_* tuning knobs).
set -euo pipefail
PROFILE=16gb; MODEL=/model; DRAFTER=; MTP=; IMAGES=; TP=1; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --mtp) MTP=$2; shift 2;;
    --images) IMAGES=$2; shift 2;;
    --tp) TP=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done
setdef() { [ -n "${!1:-}" ] || export "$1=$2"; }
case "$TP" in 1|2|4) ;; *) echo "--tp 1, 2 or 4 (got $TP)" >&2; exit 2;; esac
# --mtp under --tp is not supported in 1.2: drop it before the profile picks its context (the --mtp context sizes are tp1's)
[ "$TP" != 1 ] && [ -n "$MTP" ] && echo "--tp $TP: --mtp ignored (MTP drafting is tp1 only in 1.2)" >&2 && MTP=
case "$PROFILE" in
  16gb) # image input on: the encode workspace (0.37 GB at 4.19 MP) comes out of the KV pool -> context ~225K
        # --mtp (1.2): the head's 0.22 GB weights + 0.07 GB rings / buffers come out of the KV pool too: 242.7K pool rows
        # at 16 GiB with images off (measured: 219K document + 12 turns, 0.32 GB left under the cap) -> context ~241K
        # (KV pool 251K rows, serve plan). With images on as well the pool would be 148K rows: --mtp is ignored there.
        if [ "${IMAGES:-off}" = on ]; then W16=${AGENS_WCTX:-229380}
          [ -n "$MTP" ] && echo "16gb profile with --images on: --mtp ignored (it would cut the context to ~145K)" >&2; MTP=
        else W16=${AGENS_WCTX:-262148}; [ -n "$MTP" ] && W16=${AGENS_WCTX:-241668}; fi
        ARGS=(--budget-gib 16 --wctx "$W16" --mode personal --no-fork 1); setdef AGENS_KV_INT8_DENSE 1;;
  12gb) ARGS=(--budget-gib 12 --wctx "${AGENS_WCTX:-65540}" --mode personal --no-fork 1 --spec-copy 1)
        setdef AGENS_KV_INT8_DENSE 1; setdef AGENS_SPEC_FAR_CACHE 0  # no far-branch union cache (saves ~140 MB)
        [ "${AGENS_12GB_DRAFTER:-0}" = 1 ] || DRAFTER=  # copy-from-context drafting only: no drafter weights
        [ -n "$MTP" ] && echo "12gb profile: --mtp ignored (the head does not fit the 12 GiB plan)" >&2; MTP=;;
  # 24gb: 4 KDA state slots (serve's default is 2 x --max-batch = 8, which with the drafter leaves no KV pool at
  # 23 GiB for the 4-bit release weights: 1.0.0 failed to start "budget 23 GiB leaves no room for a KV pool")
  # --mtp with image input on: the head (0.35 GB) + the encode workspace leave a 225K-row KV pool -> context 221K
  24gb) W24=262148; [ -n "$MTP" ] && [ "${IMAGES:-}" != off ] && W24=221188
        ARGS=(--budget-gib 23 --wctx "${AGENS_WCTX:-$W24}" --mode server --max-batch 4 --max-convs 4 --sessions 4);;
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
[ -n "$MTP" ] && ARGS+=(--mtp "$MTP")
case "$TP" in
  1) ;;
  2|4) ARGS+=(--tp "$TP"); setdef AGENS_TP_LOG_DIR /tmp;;  # follower ranks' logs: /tmp/tp_rank<r>.log
esac
exec agens serve "$MODEL" --host 0.0.0.0 --port 30000 "${ARGS[@]}" "${EXTRA[@]}"
