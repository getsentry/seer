#!/bin/bash

if [ "$CELERY_WORKER_ENABLE" = "true" ]; then
    echo "Starting async worker..."
    exec python src/seer/tasks.py
else
    echo "Celery worker is disabled via environment variable CELERY_WORKER_ENABLE"
    exit 0
fi
