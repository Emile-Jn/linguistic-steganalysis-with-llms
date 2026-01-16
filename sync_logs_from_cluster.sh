#!/usr/bin/env bash

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/Linguistic-Steganalysis-with-LLMs/logs"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR="Linguistic-Steganalysis-with-LLMs/logs"

# Pull all new changes from logs directory in slurm cluster to local device (not the other way around) using rsync
rsync -avP -e ssh \
  --exclude-from='.rsyncignore' \
  "$REMOTE:$REMOTE_DIR/" \
  "$LOCAL_DIR/"