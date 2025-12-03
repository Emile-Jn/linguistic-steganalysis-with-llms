#!/usr/bin/env bash

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/entropy-steering-steganography"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR=""

# Synchronize files on cluster from local device using rsync
rsync -avP -e ssh \
  --exclude-from='.rsyncignore' \
  "$LOCAL_DIR" \
  "$REMOTE:$REMOTE_DIR"