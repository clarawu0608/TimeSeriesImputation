#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Define hyperparameter search space
seq_len=(512)
token_sizes=(128)
token_overlap_fractions=(0)
lms=(15)
missing_rates=(0.2)
nhead=(8)
num_layers=(5)
batch_size=(32)
loss_r=(0.75)
dropout=(0.2)

# Loop over 20 input files
for i in $(seq 1 3); do
  input_file="dataset/p01/p010013/csv_output/clinic_${i}.csv"

  # Loop through all combinations
  for seq_len in "${seq_len[@]}"; do
    for token_size in "${token_sizes[@]}"; do
      for frac in "${token_overlap_fractions[@]}"; do
        # Calculate integer token_overlap
        token_overlap=$(awk "BEGIN {printf \"%d\", int($frac * $token_size)}")
        for lm in "${lms[@]}"; do
          for missing_rate in "${missing_rates[@]}"; do
            for nhead in "${nhead[@]}"; do
              for loss_r in "${loss_r[@]}"; do
                for dropout in "${dropout[@]}"; do
                  for num_layer in "${num_layers[@]}"; do
                    for batch in "${batch_size[@]}"; do

                      # Define unique output directory for each experiment
                      output_dir="outputs/clinic/file${i}/s${seq_len}_t${token_size}_o${token_overlap}_lm${lm}_mr${missing_rate}_nh${nhead}_nl${num_layer}_bs${batch}_lr${loss_r}_dr${dropout}/"

                      echo "Running on file $input_file"
                      echo "  -> Output: $output_dir"

                      # Run the experiment
                      python -u run_multiscale.py \
                        --data_path "$input_file" \
                        --output_path "$output_dir" \
                        --batch_size "$batch" \
                        --seq_len "$seq_len" \
                        --token_size "$token_size" \
                        --nhead "$nhead" \
                        --num_layers "$num_layer" \
                        --token_overlap "$token_overlap" \
                        --missing_type 1 \
                        --lm "$lm" \
                        --missing_rate "$missing_rate" \
                        --loss_r "$loss_r" \
                        --dropout_rate "$dropout"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
