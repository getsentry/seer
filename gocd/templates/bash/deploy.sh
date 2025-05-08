#!/bin/bash

eval "$(/devinfra/scripts/regions/project_env_vars.py --region="${SENTRY_REGION}")"

/devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-deploy.py \
  --label-selector="service=seer" \
  --image="us-central1-docker.pkg.dev/sentryio/seer/image:${GO_REVISION_SEER_REPO}" \
  --container-name="seer"

/devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-deploy.py \
  --label-selector="service=seer-gpu" \
  --image="us-central1-docker.pkg.dev/sentryio/seer/image:${GO_REVISION_SEER_REPO}" \
  --container-name="seer-gpu"
