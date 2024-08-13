#!/bin/bash

BEAT_CMD="celery -A src.celery_app.tasks beat --loglevel=info $CELERY_BEAT_OPTIONS"

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
        exec watchmedo auto-restart -d src -p '*.py' --recursive -- $BEAT_CMD
    else
        exec $BEAT_CMD
    fi
else
    echo "Celery beat is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
