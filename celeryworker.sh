#!/bin/bash

# You can set the celery queue name via the CELERY_WORKER_QUEUE environment variable.
# If not set, the default queue name is "seer".
QUEUE="seer"
if [ "$CELERY_WORKER_QUEUE" != "" ]; then
    QUEUE="$CELERY_WORKER_QUEUE"
fi

# You can set the number of celery workers via the NUM_CELERY_WORKERS environment variable.
# If not set, the default number of workers is 16.
NUM_WORKERS="16"
if [ "$NUM_CELERY_WORKERS" != "" ]; then
    NUM_WORKERS="$NUM_CELERY_WORKERS"
fi

WORKER_CMD="celery -A src.celery_app.tasks worker --loglevel=info -Q $QUEUE -c $NUM_WORKERS $CELERY_WORKER_OPTIONS"

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
