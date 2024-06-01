#!/bin/bash

GUNICORN_ARGS=""

if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
    GUNICORN_ARGS="--reload"
fi

echo "PORT is $PORT"

exec gunicorn --bind :$PORT --worker-class sync --threads 1 --timeout 0 --access-logfile - src.seer.app:app $GUNICORN_ARGS
