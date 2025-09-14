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
pred_len=(128)  # Added pred_len to search space

# Loop over 20 input files
for i in $(seq 1 5); do
  input_file="dataset/test_simulation/high_correlation_5col/file_${i}.csv"

  # Loop through all combinations
  for seq_len_val in "${seq_len[@]}"; do
    for token_size in "${token_sizes[@]}"; do
      for frac in "${token_overlap_fractions[@]}"; do
        # Calculate integer token_overlap
        token_overlap=$(awk "BEGIN {printf \"%d\", int($frac * $token_size)}")
        for lm in "${lms[@]}"; do
          for missing_rate in "${missing_rates[@]}"; do
            for nhead_val in "${nhead[@]}"; do
              for loss_r_val in "${loss_r[@]}"; do
                for dropout_val in "${dropout[@]}"; do
                  for num_layer in "${num_layers[@]}"; do
                    for batch in "${batch_size[@]}"; do
                      for pred_len_val in "${pred_len[@]}"; do

                        # Define unique output directory for each experiment
                        output_dir="outputs/simulation/univariate/high_correlation/file${i}/s${seq_len_val}_t${token_size}_o${token_overlap}_lm${lm}_mr${missing_rate}_nh${nhead_val}_nl${num_layer}_bs${batch}_lr${loss_r_val}_dr${dropout_val}_pl${pred_len_val}/"

                        echo "Running on file $input_file"
                        echo "  -> Output: $output_dir"

                        # Run the experiment
                        python -u run_multiscale.py \
                          --data_path "$input_file" \
                          --output_path "$output_dir" \
                          --batch_size "$batch" \
                          --seq_len "$seq_len_val" \
                          --token_size "$token_size" \
                          --nhead "$nhead_val" \
                          --num_layers "$num_layer" \
                          --token_overlap "$token_overlap" \
                          --missing_type 1 \
                          --pred_len "$pred_len_val" \
                          --lm "$lm" \
                          --missing_rate "$missing_rate" \
                          --loss_r "$loss_r_val" \
                          --dropout_rate "$dropout_val"
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
done