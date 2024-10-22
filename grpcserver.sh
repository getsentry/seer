#!/bin/bash

BOOTSTRAP_BIN="/usr/local/td-grpc-bootstrap"
CMD="python -m seer.grpc"

if [ ! -f $BOOTSTRAP_BIN ]; then
    echo "Bootstrap command not found!"
    exit 0
fi
${BOOTSTRAP_BIN} --config-mesh-experimental "${MESH_ID}" > "${GRPC_XDS_BOOTSTRAP}"

if [ "$GRPC_SERVER_ENABLE" = "true" ]; then
    if [ "$DEV" = "true" ] || [ "$DEV" = "1" ]; then
        exec watchmedo auto-restart -d src -p '*.py' --recursive -- "${CMD}"
    else
        exec $CMD
    fi
else
    echo "GRPC Server is disabled via GRPC_SERVER_ENABLE variable"
    exit 0
fi
