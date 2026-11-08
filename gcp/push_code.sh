#!/bin/bash
# Bundle the code + a parsed board problem and upload to GCS.
# usage: gcp/push_code.sh [path/to/problem.json]   (default: out/problem.json inside this repo)
set -e
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
PROBLEM=${1:-out/problem.json}
STAGE=$(mktemp -d)
mkdir -p "$STAGE/eggplacer/out"
cp -r eggpcb configs scripts gcp tests "$STAGE/eggplacer/"
[ -d third_party ] && cp -r third_party "$STAGE/eggplacer/"
cp "$PROBLEM" "$STAGE/eggplacer/out/problem.json"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
tar czf /tmp/eggplacer.tgz -C "$STAGE" eggplacer
rm -rf "$STAGE"
gcloud storage cp /tmp/eggplacer.tgz $BUCKET/code/eggplacer.tgz
