#!/usr/bin/env bash

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/Linguistic-Steganalysis-with-LLMs"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR="Linguistic-Steganalysis-with-LLMs"

# Pull all new changes from logs directory in slurm cluster to local device (not the other way around) using rsync
rsync -avP -e ssh \
  --exclude-from='.rsyncignore' \
  "$REMOTE:$REMOTE_DIR/logs/" \
  "$LOCAL_DIR/logs/"

# Sync env
rsync -avP -e ssh \
  --include='pyproject.toml' \
  --include='uv.lock' \
  --exclude='*' \
  "$REMOTE:$REMOTE_DIR/" \
  "$LOCAL_DIR/"