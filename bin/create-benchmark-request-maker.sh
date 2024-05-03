#!/bin/bash

gcloud compute instances create seer-benchmark-request-maker-20240430-213423 \
    --project=ml-ai-420606 \
    --zone=us-central1-a \
    --machine-type=c3d-standard-4 \
    --network-interface=network-tier=PREMIUM,nic-type=GVNIC,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=996102297610-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=seer-benchmark-request-maker,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20240307b,mode=rw,size=200,type=projects/ml-ai-420606/zones/us-central1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any


# run this stuff then u good to go
# gcloud compute ssh    seer-benchmark-request-maker-20240430-213423
# git clone https://github.com/getsentry/seer.git
# sudo apt update
# sudo apt -y upgrade
# sudo apt install -y python3-pip
# pip3 install locust
