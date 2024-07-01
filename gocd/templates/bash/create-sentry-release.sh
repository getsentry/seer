#!/bin/bash

echo "running create-sentry-release" \
  && eval "$(/devinfra/scripts/regions/project_env_vars.py --region="${SENTRY_REGION}")" \
  && /devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-spawn-job.py \
  --container-name="seer" \
  --label-selector="service=seer" \
  "seer-create-sentry-release" \
  "us-central1-docker.pkg.dev/sentryio/seer/image:${GO_REVISION_SEER_REPO}" \
  -- \
  python \
  -m \
  seer.scripts.create_sentry_release \
  --sentry-org sentry \
  --sentry-project seer \
  --sha ${GO_REVISION_SEER_REPO}
