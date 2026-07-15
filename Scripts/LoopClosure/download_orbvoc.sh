#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-Model}
ARCHIVE="$OUTPUT_DIR/ORBvoc.txt.tar.gz"
VOCABULARY="$OUTPUT_DIR/ORBvoc.txt"
URL="https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/v1.0-release/Vocabulary/ORBvoc.txt.tar.gz"

mkdir -p "$OUTPUT_DIR"
if [[ -e "$ARCHIVE" || -e "$VOCABULARY" ]]; then
  echo "Refusing to overwrite an existing ORBvoc artifact in $OUTPUT_DIR" >&2
  exit 1
fi

wget --https-only --output-document "$ARCHIVE" "$URL"
tar -xzf "$ARCHIVE" -C "$OUTPUT_DIR"
sha256sum "$ARCHIVE" "$VOCABULARY"

echo "Source: ORB_SLAM3 v1.0-release (0df83dde1c85c7ab91a0d47de7a29685d046f637)"
echo "Training-data provenance is not authoritatively documented; record it as unverified."
