#!/bin/bash

GUNICORN_ARGS=""

if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
    GUNICORN_ARGS="--reload"
fi

exec gunicorn --bind :$PORT --worker-class sync --threads 1 --timeout 0 --logger-class seer.logging.GunicornHealthCheckFilterLogger --access-logfile - 'seer.app:start_app()' $GUNICORN_ARGS
