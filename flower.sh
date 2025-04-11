#!/bin/bash

if [ "$DEV" = "1" ]; then
    celery --broker=$CELERY_BROKER_URL flower --port=5555
else
    echo "Flower is disabled when not in DEV"
    exit 0
fi
