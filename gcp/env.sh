# Shared settings for all GCP scripts (sourced). Edit these for your project.
export PROJECT=${PROJECT:-your-gcp-project}
export REGION=${REGION:-us-central1}
export ZONE=${ZONE:-us-central1-a}           # a zone that offers NVIDIA L4 (g2-standard-*)
export BUCKET=${BUCKET:-gs://your-bucket-eggplacer}
export LABELS=project=eggplacer
