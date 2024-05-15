#!/bin/bash

/devinfra/scripts/checks/googlecloud/checkcloudbuild.py \
  "${GO_REVISION_SEER_REPO}" \
  sentryio \
  "us-central1-docker.pkg.dev/sentryio/seer/image"
