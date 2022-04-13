#This is a template of docker images for RL module
#Based on nvidia/cuda 11.3.0 base image
FROM nvidia/cuda:11.3.0-base-ubuntu20.04

LABEL maintainer="tim.su@tum.de" 

# Setting working directory to $TUM_NRP_DIR
WORKDIR /tum_nrp

# Updating the system and configuring apt-get 
RUN apt-get update && apt-get install -y --no-install-recommends 

# Installing pip and installing vim for later editing inside container 
RUN apt-get install -y python3-pip \
    vim 

#Setting up the application directory
RUN mkdir -p /tum_nrp/rl

#Copying the requirements.txt /home/rl
COPY requirements.txt /tum_nrp/rl

#Installing packages from requirements.txt file
RUN pip3 install -r /tum_nrp/rl/requirements.txt

#Installing gRPC-related packages and tools
RUN pip3 install --upgrade protobuf
RUN apt install -y iputils-ping
RUN apt install -y iproute2

#Installing the git and stable baselines 3
RUN apt-get update
RUN apt-get -y install git
RUN pip install git+https://github.com/DLR-RM/stable-baselines3

#Exposing port 6000
EXPOSE 6000
#Expose defualt jupyter notebook port
EXPOSE 8888 
#Expose default tensorboard port
EXPOSE 6006

#Starting up by running the bash
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
CMD bash
