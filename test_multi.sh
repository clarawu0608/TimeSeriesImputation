#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Define hyperparameter search space
seq_len=(512)
token_sizes=(128)
token_overlap_fractions=(0)
lms=(15)
missing_rates=(0.2 0.3 0.4)
nhead=(1)
loss_r=(0.75)
dropout=(0.1)

# Loop through all combinations
for seq_len in "${seq_len[@]}"; do
  for token_size in "${token_sizes[@]}"; do
    for frac in "${token_overlap_fractions[@]}"; do
      # Calculate integer token_overlap by flooring the value
      token_overlap=$(awk "BEGIN {printf \"%d\", int($frac * $token_size)}")
      for lm in "${lms[@]}"; do
        for missing_rate in "${missing_rates[@]}"; do
          for nhead in "${nhead[@]}"; do
            for loss_r in "${loss_r[@]}"; do
              for dropout in "${dropout[@]}"; do
              
                # Define output directory to keep track of experiments
                output_dir="outputs/multivariate/traffic/s${seq_len}_t${token_size}_o${token_overlap}_lm${lm}_mr${missing_rate}_nh${nhead}_lr${loss_r}_dr${dropout}/"

                echo "Running: seq_len=$seq_len, token_size=$token_size, token_overlap=$token_overlap, lm=$lm, missing_rate=$missing_rate, nhead=$nhead, loss_r=$loss_r, dropout=$dropout" 

                # Run the experiment
                python -u run_multivariate.py \
                  --data_path "dataset/traffic_smaller2.csv" \
                  --output_path "$output_dir" \
                  --batch_size 16 \
                  --seq_len "$seq_len" \
                  --token_t_size "$token_size" \
                  --nhead "$nhead" \
                  --token_t_overlap "$token_overlap" \
                  --token_d_size 20 \
                  --token_d_overlap 0 \
                  --missing_type 1 \
                  --lm "$lm" \
                  --missing_rate "$missing_rate" \
                  --loss_r "$loss_r" \
                  --dropout "$dropout" 
              done
            done
          done
        done
      done
    done
  done
done
