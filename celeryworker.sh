#!/bin/bash

# TODO: Remove debug log level once celery debugging is done
WORKER_CMD="celery -A src.celery_app.tasks worker --loglevel=info -c $CELERY_WORKER_CONCURRENCY $CELERY_WORKER_OPTIONS"
DEV=false

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
