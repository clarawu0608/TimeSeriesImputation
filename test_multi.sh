#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Define hyperparameter search space
seq_len=(512)
token_sizes=(128)
token_overlap_fractions=(0)
lms=(15)
missing_rates=(0.2)
nhead=(4)
loss_r=(0.75)
dropout=(0.2)
batch_sizes=(16)
num_layers_list=(5)
pred_len=(192)  # Added prediction lengths to search space

# Loop through all combinations
for seq_len_val in "${seq_len[@]}"; do
  for token_size in "${token_sizes[@]}"; do
    for frac in "${token_overlap_fractions[@]}"; do
      # Calculate integer token_overlap by flooring the value
      token_overlap=$(awk "BEGIN {printf \"%d\", int($frac * $token_size)}")
      for lm in "${lms[@]}"; do
        for missing_rate in "${missing_rates[@]}"; do
          for nhead_val in "${nhead[@]}"; do
            for loss_r_val in "${loss_r[@]}"; do
              for dropout_val in "${dropout[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                  for num_layers in "${num_layers_list[@]}"; do
                    for pred_len_val in "${pred_len[@]}"; do

                      # Define output directory to keep track of experiments
                      output_dir="outputs/prediction_mask/multivariate/multi_model/clinic/s${seq_len_val}_t${token_size}_o${token_overlap}_lm${lm}_mr${missing_rate}_nh${nhead_val}_lr${loss_r_val}_dr${dropout_val}_bs${batch_size}_nl${num_layers}_pl${pred_len_val}/"

                      echo "Running: seq_len=$seq_len_val, token_size=$token_size, token_overlap=$token_overlap, lm=$lm, missing_rate=$missing_rate, nhead=$nhead_val, loss_r=$loss_r_val, dropout=$dropout_val, batch_size=$batch_size, num_layers=$num_layers, pred_len=$pred_len_val"
                      echo "Output directory: $output_dir"

                      # Run the experiment
                      python -u run_multivariate_2d.py \
                        --data_path "dataset/p01/p010013/csv_output/clinic.csv" \
                        --output_path "$output_dir" \
                        --batch_size "$batch_size" \
                        --seq_len "$seq_len_val" \
                        --token_t_size "$token_size" \
                        --nhead "$nhead_val" \
                        --num_layers "$num_layers" \
                        --token_t_overlap "$token_overlap" \
                        --token_d_size 3 \
                        --token_d_overlap 0 \
                        --missing_type 1 \
                        --pred_len "$pred_len_val" \
                        --lm "$lm" \
                        --missing_rate "$missing_rate" \
                        --loss_r "$loss_r_val" \
                        --dropout "$dropout_val"

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