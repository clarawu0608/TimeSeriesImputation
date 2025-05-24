#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Define hyperparameter search space
seq_len=(512)
token_sizes=(128)
token_overlap_fractions=(0)
lms=(15)
missing_rates=(0.2)
nhead=(1)
loss_r=(0.75)
dropout=(0.1)

# Loop over 20 input files
for i in $(seq 1 20); do
  input_file="dataset/traffic/traffic_20col/traffic_single_${i}.csv"

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

                  # Define unique output directory for each experiment
                  output_dir="outputs/missEmbedding_test/traffic_20col/file${i}/s${seq_len}_t${token_size}_o${token_overlap}_lm${lm}_mr${missing_rate}_nh${nhead}_lr${loss_r}_dr${dropout}/"

                  echo "Running on file $input_file"
                  echo "  -> Output: $output_dir"

                  # Run the experiment
                  python -u run_test.py \
                    --data_path "$input_file" \
                    --output_path "$output_dir" \
                    --batch_size 16 \
                    --seq_len "$seq_len" \
                    --token_size "$token_size" \
                    --nhead "$nhead" \
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
