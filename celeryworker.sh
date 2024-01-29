#!/bin/bash

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    exec celery -A src.celery_app.tasks worker --loglevel=debug -c 4 # 4 workers for now because we don't acquire locks on the embedding files
else
    echo "Celery worker is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
