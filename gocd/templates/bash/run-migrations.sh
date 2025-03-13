#!/bin/bash

if [ "${SENTRY_REGION}" != "customer-1" ] && [ "${SENTRY_REGION}" != "customer-2" ] && [ "${SENTRY_REGION}" != "customer-4" ] && [ "${SENTRY_REGION}" != "customer-7" ]; then
  echo "running flask db upgrade" \
    && eval "$(/devinfra/scripts/regions/project_env_vars.py --region="${SENTRY_REGION}")" \
    && /devinfra/scripts/k8s/k8stunnel \
    && /devinfra/scripts/k8s/k8s-spawn-job.py \
    --container-name="seer-gpu" \
    --label-selector="service=seer-gpu" \
    -e="IS_DB_MIGRATION=true" \
    "seer-run-migrations" \
    "us-central1-docker.pkg.dev/sentryio/seer/image:${GO_REVISION_SEER_REPO}" \
    flask \
    db \
    upgrade
else
  echo "running flask db upgrade" \
    && eval "$(/devinfra/scripts/regions/project_env_vars.py --region="${SENTRY_REGION}")" \
    && /devinfra/scripts/k8s/k8stunnel \
    && /devinfra/scripts/k8s/k8s-spawn-job.py \
    --container-name="seer" \
    --label-selector="service=seer" \
    -e="IS_DB_MIGRATION=true" \
    "seer-run-migrations" \
    "us-central1-docker.pkg.dev/sentryio/seer/image:${GO_REVISION_SEER_REPO}" \
    flask \
    db \
    upgrade
fi
