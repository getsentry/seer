#!/bin/bash

docker build . -t us-central1-docker.pkg.dev/ml-ai-420606/seer-benchmar/benchmark:latest  --no-cache=true --platform=linux/amd64
docker image push us-central1-docker.pkg.dev/ml-ai-420606/seer-benchmar/benchmark:latest
