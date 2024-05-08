#!/bin/bash

gcloud compute ssh --zone "us-central1-a" "seer-benchmark-request-maker-20240430-213423" --project "ml-ai-420606" --command "cd /home/jferge/seer/benchmark && /home/jferge/.local/bin/locust --headless --host http://10.128.0.100 -u 1"
