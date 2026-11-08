#!/bin/bash
# bundle the code + parsed board problem and upload to GCS
set -e
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
tar czf /tmp/eggplacer.tgz --exclude='out/*.png' --exclude='out/sa_*' --exclude='logs' --exclude='__pycache__' \
    --transform 's,^\.,eggplacer,' ./eggpcb ./configs ./scripts ./third_party ./gcp ./out/problem.json ./tests
gcloud storage cp /tmp/eggplacer.tgz $BUCKET/code/eggplacer.tgz
