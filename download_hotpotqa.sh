#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/hotpotqa}"
FORCE="${2:-}"
URL="https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
OUT_PATH="${OUT_DIR}/hotpot_dev_distractor_v1.json"

mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_PATH}" && "${FORCE}" != "--force" ]]; then
  echo "File already exists: ${OUT_PATH}"
  echo "Use --force to re-download."
  exit 0
fi

echo "Downloading HotpotQA dev distractor set..."
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${OUT_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUT_PATH}" "${URL}"
else
  echo "Error: curl or wget is required."
  exit 1
fi

echo "Saved to: ${OUT_PATH}"
