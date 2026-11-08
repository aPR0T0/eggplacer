#!/bin/bash
# usage: gcp/launch.sh NAME MACHINE_TYPE ROLE(cpu|gpu) MAX_HOURS "JOB COMMAND"
# On-demand VM (user asked for no Spot) with a hard max run duration, cloud-platform scope, labels.
set -e
source "$(dirname "$0")/env.sh"
NAME=$1; MT=$2; ROLE=$3; HOURS=$4; JOB=$5
python3 "$(dirname "$0")/ledger.py" check "$MT" "$HOURS"
if [ "$ROLE" = gpu ]; then
  IMG="--image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 --image-project=deeplearning-platform-release"
  ACC="--accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE"
  DISK=100
else
  IMG="--image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud"; ACC=""; DISK=50
fi
gcloud compute instances create "$NAME" --project=$PROJECT --zone=$ZONE --machine-type="$MT" $IMG $ACC \
  --boot-disk-size=${DISK}GB --boot-disk-type=pd-balanced --scopes=cloud-platform --labels=$LABELS \
  --provisioning-model=STANDARD --max-run-duration=${HOURS}h --instance-termination-action=DELETE \
  --metadata=egg-role=$ROLE,egg-bucket=$BUCKET,install-nvidia-driver=True \
  --metadata-from-file=startup-script="$(dirname "$0")/startup.sh",egg-job=<(echo "$JOB")
python3 "$(dirname "$0")/ledger.py" start "$NAME" "$MT" "$HOURS"
