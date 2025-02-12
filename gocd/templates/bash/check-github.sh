#!/bin/bash

/devinfra/scripts/checks/githubactions/checkruns.py \
  getsentry/seer \
  "${GO_REVISION_SEER_REPO}" \
  "Parallelized Tests"
