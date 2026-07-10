#!/usr/bin/env bash
set -euo pipefail

ODOM_CONFIG="${ODOM_CONFIG:-Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml}"
DATA_CONFIG="${DATA_CONFIG:-Config/Sequence/TartanAir_example.yaml}"
TRAIN_ROOT="${TRAIN_ROOT:-./Results/train_cache}"
FINAL_ROOT="${FINAL_ROOT:-./Results/loop_phase_b}"
VOCAB_PATH="${VOCAB_PATH:-Model/ORB_BoW_4096.npz}"

if [[ -f "$VOCAB_PATH" ]]; then
  echo "[PhaseB-Test] Vocabulary already exists: $VOCAB_PATH"
  echo "[PhaseB-Test] Skip cache-only run and vocabulary training. Remove it manually if you need a fresh vocabulary."
else
  echo "[PhaseB-Test] 1/4 Run MAC-VO once without vocabulary to build loop cache."
  python3 MACVO.py --odom "$ODOM_CONFIG" --data "$DATA_CONFIG" --resultRoot "$TRAIN_ROOT"

  echo "[PhaseB-Test] 2/4 Locate newest loop_closure cache under $TRAIN_ROOT."
  CACHE_DIR=$(find "$TRAIN_ROOT" -path "*/loop_closure/index.json" -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -z "${CACHE_DIR:-}" ]]; then
    echo "[PhaseB-Test][ERROR] No loop_closure/index.json found under $TRAIN_ROOT" >&2
    exit 1
  fi
  echo "[PhaseB-Test] Cache directory: $CACHE_DIR"

  echo "[PhaseB-Test] 3/4 Train ORB-BoW vocabulary."
  mkdir -p "$(dirname "$VOCAB_PATH")"
  python3 Scripts/LoopClosure/train_orb_vocabulary.py "$CACHE_DIR" --output "$VOCAB_PATH"
fi

if [[ ! -s "$VOCAB_PATH" ]]; then
  echo "[PhaseB-Test][ERROR] Vocabulary missing or empty: $VOCAB_PATH" >&2
  exit 1
fi

echo "[PhaseB-Test] Vocabulary OK: $VOCAB_PATH"

echo "[PhaseB-Test] 4/4 Run MAC-VO PhaseB with vocabulary and timing."
python3 MACVO.py --odom "$ODOM_CONFIG" --data "$DATA_CONFIG" --resultRoot "$FINAL_ROOT" --timing
