#!/bin/bash

gcloud compute instances create-with-container seer-benchmark-1 \
    --project=ml-ai-420606 \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=count=1,type=nvidia-l4 \
    --network-interface=network-tier=PREMIUM,nic-type=GVNIC,subnet=default,private-network-ip=10.128.0.100 \
    --provisioning-model=STANDARD \
    --maintenance-policy=TERMINATE \
    --service-account=996102297610-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --tags=http-server \
    --image=projects/cos-cloud/global/images/cos-stable-109-17800-147-60 \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --boot-disk-device-name=seer-benchmark \
    --container-image=us.gcr.io/sentryio/seer@sha256:84e5d226dd9b95628d441e944403ecd45740fb1deec22c4c6d89f3baf8035c6c \
    --container-restart-policy=always \
    --container-privileged \
    --container-stdin \
    --container-tty \
    --container-env=SEVERITY_ENABLED=true,GROUPING_ENABLED=true,AUTOFIXABILITY_SCORING_ENABLED=true,PORT=80,SENTRY_BASE_URL=values.sentry_base_url,DATABASE_URL=postgresql\+psycopg://root:seer@10.128.0.7/seer \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud,container-vm=cos-stable-109-17800-147-60
