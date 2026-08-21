#!/usr/bin/env bash
# Phase 8 -- pushes the ONNX serving image to Artifact Registry and deploys
# it to Cloud Run.
#
# Backend: ONNX Runtime CPU (Dockerfile.onnx), chosen over PyTorch based on
# scripts/benchmark_phase8.py's measured results -- 2.4x smaller image
# (131.9 MB vs 319.4 MB), faster cold start (2.03s vs 2.29s), ~7% slower
# per-request inference (3000.2ms vs 2781.3ms). See the Phase 8 plan.md
# entry for the full writeup; this script does not re-litigate that choice.
#
# --- One-time setup (run manually before first use, NOT by this script) ---
#   gcloud auth login
#   gcloud config set project "$PROJECT_ID"
#   gcloud services enable artifactregistry.googleapis.com run.googleapis.com
#   gcloud artifacts repositories create "$REPO_NAME" \
#       --repository-format=docker --location="$REGION"
#   gcloud auth configure-docker "${REGION}-docker.pkg.dev"
# Deliberately left as documented manual steps rather than scripted: they're
# one-time and idempotency-fragile (re-running `repositories create` against
# an existing repo errors rather than no-ops), so silently wrapping them
# risks masking a real problem behind a "already exists" error on a rerun.
#
# Usage:
#   PROJECT_ID=my-gcp-project ./scripts/deploy_phase8.sh
#
# Cost note, read before running with --allow-unauthenticated (below):
# this makes the endpoint publicly callable, and each /enhance call is a
# real ~3s of billed CPU time. --max-instances caps worst-case concurrent
# cost exposure but does NOT cap total spend if it sits at max for a while --
# set a GCP budget alert (Billing > Budgets & alerts) before sharing the URL
# widely, this script does not do that for you.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-project ./scripts/deploy_phase8.sh}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-low-light-enhancement}"
SERVICE_NAME="${SERVICE_NAME:-low-light-enhancement}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/serve-onnx:${IMAGE_TAG}"

echo "==> Building ${IMAGE_URI} from Dockerfile.onnx"
docker build -f Dockerfile.onnx -t "${IMAGE_URI}" .

echo "==> Pushing to Artifact Registry"
docker push "${IMAGE_URI}"

echo "==> Deploying to Cloud Run"
# --memory=2Gi is Cloud Run's enforced minimum for --cpu=4 (confirmed by a
# real deploy failure at 1Gi/4cpu: "For 4.0 CPU, memory must be between 2Gi
# and 16Gi inclusive") -- not a guess. --cpu=4 IS measured: the live
# deployment at --cpu=2 ran ~25-32% slower than the local benchmark
# (3700-3965ms vs ~3000ms server inference), traced to Cloud Run's 2-vCPU
# cap vs. the 12 cores the local container saw (`docker run --rm
# low-light-enhancement:onnx nproc`). The gap was well under the 6x core
# ratio, suggesting this model saturates useful parallelism before 12
# threads -- 4 should capture most of the available speedup without
# assuming more cores keeps helping linearly.
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_URI}" \
    --platform managed \
    --region "${REGION}" \
    --memory 2Gi \
    --cpu 4 \
    --timeout 60 \
    --max-instances 3 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars=ORT_INTRA_OP_THREADS=2
# ORT_INTRA_OP_THREADS=2 matches the live revision (-00003) this script's
# config is meant to reproduce -- included for reproducibility, NOT because
# it's proven to help. scripts/benchmark_cloud_run.py's n=10 sampling
# against that revision (mean=3846.7ms, std=298.3ms) fully contains both
# single-sample reads from the --cpu=2 config it was compared against
# (3709.3ms, 3964.7ms) -- meaning the whole --cpu=2 -> --cpu=4 ->
# thread-pinning tuning sequence was likely never distinguishable from
# request-to-request noise. Kept on "no evidence it's wrong," not "proven
# better" -- see the Phase 8 plan.md entry for the full writeup before
# treating this value as settled.

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" --format='value(status.url)')

echo ""
echo "==> Deployed: ${SERVICE_URL}"
echo "==> Verify with:"
echo "    python scripts/verify_phase8.py --url ${SERVICE_URL} \\"
echo "        --low-image data/LOLdataset/eval15/low/1.png \\"
echo "        --high-image data/LOLdataset/eval15/high/1.png"
echo ""
echo "==> Check Cloud Logging with:"
echo "    gcloud logging read \\"
echo "        'resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\"' \\"
echo "        --limit 20 --freshness=10m --format='value(textPayload)'"
