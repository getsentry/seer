#!/bin/bash

/devinfra/scripts/checks/googlecloud/check_cloudbuild.py \
  sentryio \
  seer \
  seer-builder \
  "${GO_REVISION_SEER_REPO}" \
  main
