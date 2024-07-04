#!/bin/bash
set -e
sentry-cli releases new "${GO_REVISION_SEER_REPO}"
sentry-cli releases deploys "${GO_REVISION_SEER_REPO}" new -e production
sentry-cli releases finalize "${GO_REVISION_SEER_REPO}"
sentry-cli releases set-commits "${GO_REVISION_SEER_REPO}" --auto || true
