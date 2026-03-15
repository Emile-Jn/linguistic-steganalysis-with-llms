#!/usr/bin/env bash

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/Linguistic-Steganalysis-with-LLMs"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR="Linguistic-Steganalysis-with-LLMs"

# Pull all new changes from the outputs directory and the uv env in the slurm cluster to local device (not the other way around) using rsync
rsync -avP -e ssh \
  --include='outputs' \
  --include='outputs/**' \
  --include='pyproject.toml' \
  --include='uv.lock' \
  --exclude='*' \
  "$REMOTE:$REMOTE_DIR/" \
  "$LOCAL_DIR/"