#!/bin/bash

CMD="python -m seer.grpc"

if [ "$GRPC_SERVER_ENABLE" = "true" ]; then
    if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
        exec watchmedo auto-restart -d src -p '*.py' --recursive -- $CMD
    else
        exec $CMD
    fi
else
    echo "GRPC Server is disabled via GRPC_SERVER_ENABLED variable"
    exit 0
fi
