#!/bin/bash

# You can set the celery queue name via the CELERY_WORKER_QUEUE environment variable.
# If not set, the default queue name is "seer".
QUEUE="seer"
if [ "$CELERY_WORKER_QUEUE" != "" ]; then
    QUEUE="$CELERY_WORKER_QUEUE"
fi

WORKER_CMD="celery -A src.celery_app.tasks worker --loglevel=info -Q $QUEUE $CELERY_WORKER_OPTIONS"

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
        exec watchmedo auto-restart -d src -p '*.py' --recursive -- $WORKER_CMD
    else
        exec $WORKER_CMD
    fi
else
    echo "Celery worker is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
