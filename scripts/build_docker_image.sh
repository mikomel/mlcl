#!/usr/bin/env bash

IMAGE_URI="mikomel/mlcl:latest"

docker build -t ${IMAGE_URI} -f docker/nvidia.Dockerfile .
