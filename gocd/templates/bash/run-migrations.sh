#!/bin/bash

echo "running flask db upgrade" \
  && /devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-spawn-job.py \
  --container-name="seer" \
  --label-selector="service=seer" \
  "seer-run-migrations" \
  "us.gcr.io/sentryio/seer:${GO_REVISION_SEER_REPO}" \
  -- \
  flask \
  db \
  upgrade
