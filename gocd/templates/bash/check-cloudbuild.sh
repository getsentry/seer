#!/bin/bash

/devinfra/scripts/checks/googlecloud/checkcloudbuild.py \
  "${GO_REVISION_SEER_REPO}" \
  sentryio \
  "us.gcr.io/sentryio/seer"
