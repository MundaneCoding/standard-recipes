#!/bin/bash

VERSION='py3.9-torch1.13-cuda11.8.0-r0'
IMAGE='agilemelon/ddpm'
LOCAL_IMAGE='ddpm'

MOUNT_OPS="-v $(pwd):/workspace"

OPTS="--user $UID"
MEM_OPT="--ipc=host --ulimit memlock=-1"

if [ $# -eq 0 ]; then
    docker run -it --rm --gpus all $MOUNT_OPS $OPTS $MEM_OPT $IMAGE:$VERSION
elif [ "$1" = "cpu" ]; then
    docker run -it --rm $MOUNT_OPS $OPTS $IMAGE:$VERSION
elif [ "$1" = "local" ]; then
    docker run -it --rm --gpus all $MOUNT_OPS $OPTS $MEM_OPT $LOCAL_IMAGE:latest
elif [ "$1" = "local_cpu" ]; then
    docker run -it --rm $MOUNT_OPS $OPTS $LOCAL_IMAGE:latest
elif [ "$1" = "build" ]; then
    # this is just a template
    docker build -t $LOCAL_IMAGE -f ./docker/Dockerfile ./docker
else
    echo "Unknown argument: $1"
fi

## template for building and pushing
# docker build -t $LOCAL_IMAGE -f ./docker/Dockerfile ./docker
# docker tag $LOCAL_IMAGE something:version
# docker push something:version
