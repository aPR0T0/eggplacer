#!/bin/bash
# Fetch the ChipDiffusion denoiser backbone (Lee et al., arXiv:2407.12282) into third_party/.
# It is not redistributed in this repository because upstream has no licence file.
set -euo pipefail
COMMIT=6973e90ee1afc345b56da486bbf7dc9772ada1a4
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/chipdiffusion"
TMP=$(mktemp -d)
git clone -q https://github.com/vint-1/chipdiffusion.git "$TMP/cd"
git -C "$TMP/cd" checkout -q "$COMMIT"
mkdir -p "$DEST"
cp -r "$TMP/cd/diffusion/networks" "$TMP/cd/diffusion/pos_encoding.py" "$DEST/"
echo "$COMMIT" > "$DEST/UPSTREAM_COMMIT"
rm -rf "$TMP"
echo "ChipDiffusion backbone at $COMMIT -> $DEST"
