#!/bin/bash

eval "$(/devinfra/scripts/regions/project_env_vars.py --region="${SENTRY_REGION}")"

/devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-deploy.py \
  --label-selector="service=seer" \
  --image="us.gcr.io/sentryio/seer:${GO_REVISION_SEER_REPO}" \
  --container-name="seer"

/devinfra/scripts/k8s/k8stunnel \
  && /devinfra/scripts/k8s/k8s-deploy.py \
  --label-selector="service=seer-autofix" \
  --image="us.gcr.io/sentryio/seer:${GO_REVISION_SEER_REPO}" \
  --container-name="seer-autofix"
