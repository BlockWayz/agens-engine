#!/bin/bash
# Container entry point: --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [--mtp DIR] [--images on|off]
#                        [--tp 1|2|4] [-- extra agens serve flags]
# --mtp: the model's MTP head (docs/mtp.md; mount it like the drafter, e.g. -v /path/mtp:/mtp:ro --mtp /mtp). It joins the
# drafting chain (context copy, MTP, DFlash2 drafter) in the 16gb, 24gb and 48gb profiles; the 12gb profile ignores it.
# --tp N: tensor parallel over the container's first N GPUs (one process per GPU; `docker run --gpus '"device=0,1"'
# --ipc=host`). The profile's budget is per GPU. Under --tp the KV cache stays on the GPUs (each holds 1 / N of the K/V
# heads; no host tiering), so the per-GPU plan has more context or more conversation slots than tp1 on the same card.
# --mtp with --tp 2|4 (1.3): MTP joins the drafting chain on every rank; the context is the profile's full --wctx, clamped by
# the engine to the per-GPU KV pool (logged).
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
# 1.3.6: the profiles live in the binary (`agens serve --profile`, crates/agens-cli/src/profile.rs) so `agens serve` run
# outside the container gets the same flags and defaults; this script maps its options onto it. The numbers behind the
# profiles' context sizes:
#  16gb: image input on: the encode workspace (0.37 GB at 4.19 MP) comes out of the KV pool -> context ~225K.
#        --mtp (1.2): the head's 0.22 GB weights + 0.07 GB rings / buffers come out of the KV pool too: 242.7K pool rows
#        at 16 GiB with images off (measured: 219K document + 12 turns, 0.32 GB left under the cap) -> context ~241K
#        (KV pool 251K rows, serve plan). 1.3: images on + --mtp keeps MTP (the encode workspace shares the prefill
#        temporaries): KV pool 227,424 rows on the 48 GB sm89 card at the 16 GiB cap -> --wctx 221188 (context 217,092;
#        219K document + a turn + 2 images: 16.75 GB, 0.43 GB left under the cap).
#        1.3.6: AGENS_KV_MEANS_INT8=1 (int8 BCSA far-field block means, ~8.6 KB less host memory per row) is opt-in:
#        gated vs bf16 means (NIAH 32K-131K equal, GPQA 16K 76.0 vs 75.4, p 0.89; HumanEval equal) but decode at 30K
#        6-13% slower on the 48 GB sm89 card at the 16 GiB cap (105.2 -> 98.4 greedy, 87.0 -> 76.0 temp + presence).
#  12gb: no far-branch union cache (AGENS_SPEC_FAR_CACHE=0, saves ~140 MB); copy-from-context drafting only
#        (AGENS_12GB_DRAFTER=1 keeps --drafter); --mtp ignored (the head does not fit the 12 GiB plan).
#  24gb: 4 KDA state slots (serve's default is 2 x --max-batch = 8, which with the drafter leaves no KV pool at 23 GiB
#        for the 4-bit release weights: 1.0.0 failed to start "budget 23 GiB leaves no room for a KV pool").
#        --mtp with image input on (1.3): the encode workspace shares the prefill temporaries -> KV pool 288,356 rows on
#        the 48 GB sm89 card (1.2: 236,574 -> context 221K), enough for the full 262K.
ARGS=(--profile "$PROFILE")
case "${IMAGES:-}" in
  on) ARGS+=(--vision 1);;
  off) ARGS+=(--vision 0);;
  "") ;;  # the profile's default: off for 12gb / 16gb, on for 24gb / 48gb when the checkpoint has the tower
  *) echo "--images on|off (got $IMAGES)" >&2; exit 2;;
esac
[ -n "$DRAFTER" ] && ARGS+=(--spec-drafter "$DRAFTER")
[ -n "$MTP" ] && ARGS+=(--mtp "$MTP")
[ "$TP" != 1 ] && ARGS+=(--tp "$TP")
exec agens serve "$MODEL" --host 0.0.0.0 --port 30000 "${ARGS[@]}" "${EXTRA[@]}"
