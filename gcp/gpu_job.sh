#!/bin/bash
# GPU VM job: a short validation run on the first dataset as soon as it exists, then the full run.
# Override via the job string, e.g. 'SMALL=v1_5k FULL=v1_5k,v1_20k_a HOURS=6 bash gcp/gpu_job.sh'.
set -x
B=$BUCKET
SMALL=${SMALL:-v1_5k}
FULL=${FULL:-v1_5k,v1_20k_a,v1_20k_b}
SMALL_HOURS=${SMALL_HOURS:-0.75}
HOURS=${HOURS:-6}
wait_for() { until gcloud storage ls "$1/_DONE" >/dev/null 2>&1; do sleep 60; done; }
nvidia-smi
mkdir -p data
wait_for $B/data/$SMALL
gcloud storage rsync -r $B/data/$SMALL data/$SMALL
$PY -m eggpcb.train --data data/$SMALL --ckpt ckpt/small --gcs $B/ckpt/small --steps 20000 --hours $SMALL_HOURS \
    --val 250 --save_every 5000 --log_every 200
DIRS=""
for d in ${FULL//,/ }; do
  wait_for $B/data/$d
  gcloud storage rsync -r $B/data/$d data/$d
  DIRS="$DIRS,data/$d"
done
$PY -m eggpcb.train --data ${DIRS#,} --ckpt ckpt/full --gcs $B/ckpt/full \
    --steps 400000 --hours $HOURS --val 500 --save_every 10000 --log_every 500
