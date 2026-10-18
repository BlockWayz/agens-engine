#!/bin/bash
# Agens Engine: run a GPU profile (OpenAI-compatible server on port 30000).
#
#   scripts/agens-run.sh --profile 12gb|16gb|24gb|48gb --model DIR [--drafter DIR] [--gpu N] [--port P] [--image TAG]
#                        [-- extra agens serve flags]
#
# Inside the image the entry point takes the same flags (--model / --drafter are then container paths):
#   docker run --gpus device=0 --ipc=host -p 30000:30000 -v /path/model:/model:ro -v /path/drafter:/drafter:ro \
#     agens-engine:<tag> --profile 16gb --model /model --drafter /drafter
#
# Profiles (all under a hard memory cap: the engine plans weights, context and KV within the budget and holds the rest
# of the card so an overshoot fails instead of slowing down):
#   16gb  --budget-gib 16, 262K context, personal mode (one conversation), speculative decoding with the drafter,
#         KV cache tiered to host RAM (~24 GB of pinned host memory at full context), INT8 dense-layer KV.
#   12gb  --budget-gib 12, context and drafter per the measured 12 GB plan (see README), otherwise as 16gb.
#   24gb / 48gb  --budget-gib 23 / 46, server mode (batched requests and batched speculative decoding).
set -euo pipefail
PROFILE=16gb; MODEL=; DRAFTER=; GPU=0; PORT=30000; IMAGE=${AGENS_IMAGE:-ghcr.io/blockwayz/agens-engine:1.0.0}; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --image) IMAGE=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown flag $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] || { echo "--model DIR is required" >&2; exit 2; }
M=(-v "$(cd "$MODEL" && pwd)":/model:ro); D=()
[ -n "$DRAFTER" ] && M+=(-v "$(cd "$DRAFTER" && pwd)":/drafter:ro) && D=(--drafter /drafter)
exec docker run --rm --gpus "device=$GPU" --ipc=host -p "$PORT":30000 "${M[@]}" "$IMAGE" \
  --profile "$PROFILE" --model /model "${D[@]}" -- "${EXTRA[@]}"
