#!/bin/bash

gcloud compute instances create seer-benchmark-1 \
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
    --image=projects/cos-cloud/global/images/cos-stable-113-18244-1-65 \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --boot-disk-device-name=seer-benchmark \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud,container-vm=cos-stable-109-17800-147-60 \
    --metadata-from-file user-data=cloud-config.yml
