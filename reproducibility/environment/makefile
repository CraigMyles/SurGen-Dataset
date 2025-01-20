#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = SurGen-Reproduce

# docker options # must be lowercase
DOCKER_IMAGE_NAME = surgen-reproduce

# docker build process info
LOCAL_USER = $(USER)
LOCAL_UID = $(shell id -u)
LOCAL_GID = $(shell id -g)

# network
JUPYTER_PORT :=6063# TODO: change this per user

#################################################################################
# PYTHON ENVIRONMENT COMMANDS                                                   #
#################################################################################
# save the python environment so it can be recreated
export_env:
	conda env export > environment.yml

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build --build-arg LOCAL_USER=$(LOCAL_USER) \
				 --build-arg LOCAL_UID=$(LOCAL_UID) \
				 --build-arg LOCAL_GID=$(LOCAL_GID) \
				 -t $(DOCKER_IMAGE_NAME):v1 .

# note - if the data moves around, change where the volumes are mapped
docker_run:
	docker run --gpus all --shm-size 8G \
				--name $(LOCAL_USER)_$(DOCKER_IMAGE_NAME) \
				--user $(LOCAL_UID):$(LOCAL_GID) \
				-p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v /mnt/isilon1/craig:/home/$(LOCAL_USER)/data \
				-v /home/$(LOCAL_USER)/.vscode-server/extensions:/home/$(LOCAL_USER)/.vscode-server/extensions \
				-it $(DOCKER_IMAGE_NAME):v1
