#!/bin/bash

# Base directory for checkpoint files
ckpt_dir="../mase_output/jsc-tiny_classification_jsc_2024-01-26/software/training_ckpts"

# Array of checkpoint files
ckpt_files=("best.ckpt" "best-v1.ckpt" "best-v2.ckpt" "best-v3.ckpt" "best-v4.ckpt")

# Loop through each checkpoint file
for ckpt in "${ckpt_files[@]}"
do
    echo "Testing model with checkpoint file: $ckpt"
    touch "$ckpt results.txt"
    ./ch test jsc-tiny jsc --load "$ckpt_dir/$ckpt" --load-type pl >> "$ckpt results.txt"
    echo "Testing completed for checkpoint file: $ckpt"
done

echo "All model tests completed."
