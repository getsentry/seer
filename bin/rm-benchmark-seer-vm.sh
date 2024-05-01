#!/bin/bash

gcloud compute instances delete seer-benchmark-1 \
    --project=ml-ai-420606 \
    --zone=us-central1-a \
    --quiet
