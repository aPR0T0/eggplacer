#!/bin/bash
# VM startup script: fetch code from GCS, set up python, run the job given in metadata `egg-job`,
# mirror its log to GCS, then delete the VM (fallback: power off). An idle watchdog powers the VM off
# if no eggplacer process has run for 20 minutes. Re-running (reboot) resumes: jobs are resumable.
set -x
MD=http://metadata.google.internal/computeMetadata/v1/instance
get() { curl -s -H "Metadata-Flavor: Google" "$MD/attributes/$1"; }
JOB="$(get egg-job)"; BUCKET="$(get egg-bucket)"; ROLE="$(get egg-role)"
NAME=$(curl -s -H "Metadata-Flavor: Google" $MD/name); ZONE=$(curl -s -H "Metadata-Flavor: Google" $MD/zone | awk -F/ '{print $NF}')
mkdir -p /opt/egg && cd /opt/egg

cat > /opt/egg/watchdog.sh <<'WD'
#!/bin/bash
idle=0
while true; do
  # busy while setup runs or the job process (pid in /opt/egg/job.pid) is alive
  if pgrep -f "pip install|apt-get|gcloud storage cp" >/dev/null || { [ -f /opt/egg/job.pid ] && kill -0 "$(cat /opt/egg/job.pid)" 2>/dev/null; }; then idle=0; else idle=$((idle+1)); fi
  if [ $idle -ge 20 ]; then shutdown -h now; fi
  sleep 60
done
WD
chmod +x /opt/egg/watchdog.sh; nohup /opt/egg/watchdog.sh >/dev/null 2>&1 &

if [ ! -f .setup_done ]; then
  gcloud storage cp $BUCKET/code/eggplacer.tgz . && tar xzf eggplacer.tgz
  if [ "$ROLE" = gpu ]; then
    for i in $(seq 1 60); do nvidia-smi && break; sleep 10; done
    PY=$(for p in /opt/conda/bin/python /usr/bin/python3; do $p -c "import torch" 2>/dev/null && echo $p && break; done)
    $PY -m venv --system-site-packages /opt/egg/venv || (apt-get update -y && apt-get install -y python3-venv && $PY -m venv --system-site-packages /opt/egg/venv)
    /opt/egg/venv/bin/pip install -q torch_geometric numba scipy
  else
    apt-get update -y && apt-get install -y python3-venv python3-pip
    python3 -m venv /opt/egg/venv
    /opt/egg/venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    /opt/egg/venv/bin/pip install -q numpy scipy numba torch_geometric
  fi
  touch .setup_done
fi
cd /opt/egg/eggplacer
export BUCKET PY=/opt/egg/venv/bin/python
( eval "$JOB" ) >> /opt/egg/job.log 2>&1 &
JOBPID=$!
echo $JOBPID > /opt/egg/job.pid
while kill -0 $JOBPID 2>/dev/null; do gcloud storage cp -q /opt/egg/job.log $BUCKET/logs/$NAME.log; sleep 60; done
wait $JOBPID; echo "JOB EXIT $?" >> /opt/egg/job.log
gcloud storage cp -q /opt/egg/job.log $BUCKET/logs/$NAME.log
gcloud compute instances delete $NAME --zone $ZONE --quiet || shutdown -h now
