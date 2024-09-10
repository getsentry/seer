#!/bin/bash

# TODO: Remove debug log level once celery debugging is done
WORKER_CMD="celery -A src.celery_app.tasks worker --loglevel=debug -c $CELERY_WORKER_CONCURRENCY $CELERY_WORKER_OPTIONS"
DEV=false

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    exec $WORKER_CMD
else
    echo "Celery worker is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
