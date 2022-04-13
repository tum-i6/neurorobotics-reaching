#!/bin/bash

# Create volumes
docker volume create nrp_storage
docker volume create nrp_models
docker volume create nrp_experiments

# Initialize model storage
docker run --rm -v $TUM_NRP_DIR:/tum_nrp -v nrp_storage:/home/bbpnrsoa/.opt/nrpStorage -v nrp_models:/home/bbpnrsoa/nrp/src/Models -v nrp_experiments:/home/bbpnrsoa/nrp/src/Experiments hbpneurorobotics/nrp:3.1 bash "/tum_nrp/nrp-docker/deploy/setup/setup_storage.sh"

# TUM network
#docker run --rm --net=host -v $TUM_NRP_DIR:/tum_nrp -v nrp_storage:/home/bbpnrsoa/.opt/nrpStorage -v nrp_models:/home/bbpnrsoa/nrp/src/Models -v nrp_experiments:/home/bbpnrsoa/nrp/src/Experiments hbpneurorobotics/nrp:3.1 bash "/tum_nrp/nrp-docker/deploy/setup/setup_storage.sh"
