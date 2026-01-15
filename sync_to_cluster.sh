#!/usr/bin/env bash

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/Linguistic-Steganalysis-with-LLMs"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR=""

# Push all new changes from local device to slurm cluster (not the other way around) using rsync
rsync -avP -e ssh \
  --exclude-from='.rsyncignore' \
  "$LOCAL_DIR" \
  "$REMOTE:$REMOTE_DIR"