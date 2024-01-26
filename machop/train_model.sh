#!/bin/bash

# Array of learning rates
learning_rates=(1e-06 1e-07 1e-05 1e-04 1e-03)

# Loop through each learning rate
for lr in "${learning_rates[@]}"
do
    echo "Starting training with learning rate: $lr"
    ./ch train jsc-tiny jsc --max-epochs 10 --batch-size 128 --learning-rate $lr
    echo "Training completed for learning rate: $lr"
done

echo "All training runs completed."
