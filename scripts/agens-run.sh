#!/bin/bash
# Agens Engine: run a GPU profile (OpenAI-compatible server on port 30000).
#
#   scripts/agens-run.sh --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [--mtp DIR] [--gpu N] [--port P] [--image TAG]
#                        [--images on|off]   (image input; default on for 24gb / 48gb, off for 12gb / 16gb)
#                        [--tp 2|4 --gpu 0,1[,2,3]]   (tensor parallel over those GPUs; the budget is per GPU)
#                        [-- extra agens serve flags]
#
# Inside the image the entry point takes the same flags (--model / --drafter are then container paths):
#   docker run --gpus device=0 --ipc=host -p 30000:30000 -v /path/model:/model:ro -v /path/drafter:/drafter:ro \
#     [-v /path/mtp:/mtp:ro] agens-engine:<tag> --profile 16gb --model /model --drafter /drafter [--mtp /mtp]
# --mtp: the model's MTP head (1.2), a third draft source beside context copy and the drafter (not in 12gb).
#
# Profiles (all under a hard memory cap: the engine plans weights, context and KV within the budget and holds the rest
# of the card so an overshoot fails instead of slowing down):
#   16gb  --budget-gib 16, 262K context, personal mode (one conversation), speculative decoding with the drafter,
#         KV cache tiered to host RAM (~24 GB of pinned host memory at full context), INT8 dense-layer KV.
#   12gb  --budget-gib 12, 64K context, personal mode, speculative decoding from copy-from-context proposals only
#         (no drafter weights), INT8 dense-layer KV.
#   24gb / 48gb  --budget-gib 23 / 46, server mode (batched requests and batched speculative decoding).
set -euo pipefail
PROFILE=16gb; MODEL=; DRAFTER=; MTP=; GPU=0; PORT=30000; IMAGE=${AGENS_IMAGE:-ghcr.io/blockwayz/agens-engine:1.3.3}; IMAGES=(); TP=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --mtp) MTP=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --image) IMAGE=$2; shift 2;;
    --images) IMAGES=(--images "$2"); shift 2;;
    --tp) TP=(--tp "$2"); shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown flag $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] || { echo "--model DIR is required" >&2; exit 2; }
M=(-v "$(cd "$MODEL" && pwd)":/model:ro); D=()
[ -n "$DRAFTER" ] && M+=(-v "$(cd "$DRAFTER" && pwd)":/drafter:ro) && D=(--drafter /drafter)
[ -n "$MTP" ] && M+=(-v "$(cd "$MTP" && pwd)":/mtp:ro) && D+=(--mtp /mtp)
exec docker run --rm --gpus "\"device=$GPU\"" --ipc=host -p "$PORT":30000 "${M[@]}" "$IMAGE" \
  --profile "$PROFILE" --model /model "${D[@]}" "${IMAGES[@]}" "${TP[@]}" -- "${EXTRA[@]}"
