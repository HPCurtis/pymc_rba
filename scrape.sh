#!/bin/bash
aws s3 sync --no-sign-request s3://openneuro.org/ds002790 ./data/test-subject \
    --exclude "*" \
    --include "participants.json" \
    --include "dataset_description.json" \
    --include "task-stopsignal_acq-seq_events.json" \
    --include "task-stopsignal_acq-seq_bold.json"
  
  # Loop to include participants from sub-0001 to sub-0018
  for i in $(seq -w 1 18); do
    aws s3 sync --no-sign-request s3://openneuro.org/ds002790 ./data/test-subject \
      --exclude "*" \
      --include "derivatives/fmriprep/sub-00$i/func/*stopsignal_acq-seq_space-MNI152NLin2009cAsym*" \
      --include "derivatives/fmriprep/sub-00$i/func/*stopsignal_acq-seq_desc-confounds_regressors*" \
      --include "sub-00$i/func/*task-stopsignal_acq-seq_events.tsv"
  done