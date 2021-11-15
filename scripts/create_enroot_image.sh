#!/usr/bin/env bash

IMAGE_URI="mikomel/mlcl:latest"
SQSH_FILENAME="mikomel-mlcl-latest.sqsh"

enroot import --output "${SQSH_FILENAME}" "dockerd://${IMAGE_URI}"
enroot create "${SQSH_FILENAME}"
