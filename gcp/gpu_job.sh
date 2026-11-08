#!/bin/bash
# GPU VM job: small validation run on the 5k set as soon as it exists, then the full run on 45k.
set -x
B=$BUCKET
wait_for() { until gcloud storage ls "$1/_DONE" >/dev/null 2>&1; do sleep 60; done; }
nvidia-smi
wait_for $B/data/v1_5k
mkdir -p data && gcloud storage rsync -r $B/data/v1_5k data/v1_5k
$PY -m eggpcb.train --data data/v1_5k --ckpt ckpt/small --gcs $B/ckpt/small --steps 20000 --hours 0.75 \
    --val 250 --save_every 5000 --log_every 200
wait_for $B/data/v1_20k_a; wait_for $B/data/v1_20k_b
gcloud storage rsync -r $B/data/v1_20k_a data/v1_20k_a
gcloud storage rsync -r $B/data/v1_20k_b data/v1_20k_b
$PY -m eggpcb.train --data data/v1_5k,data/v1_20k_a,data/v1_20k_b --ckpt ckpt/full --gcs $B/ckpt/full \
    --steps 400000 --hours 6 --val 500 --save_every 10000 --log_every 500
