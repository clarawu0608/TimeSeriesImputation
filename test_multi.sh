#!/bin/bash
# Activate virtual environment if needed
. venv/bin/activate

# Run the script with arguments
python -u run_multivariate.py \
  --batch_size 8 \
  --seq_len 80 \
  --token_t_size 16 \
  --token_t_overlap 0 \
  --token_d_size 100 \
  --token_d_overlap 0 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3 \
  --loss_r 0.55 \

python -u run_multivariate.py \
  --batch_size 8 \
  --seq_len 80 \
  --token_t_size 16 \
  --token_t_overlap 0 \
  --token_d_size 100 \
  --token_d_overlap 0 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3 \
  --loss_r 0.6 \

python -u run_multivariate.py \
  --batch_size 8 \
  --seq_len 80 \
  --token_t_size 16 \
  --token_t_overlap 0 \
  --token_d_size 100 \
  --token_d_overlap 0 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3 \
  --loss_r 0.7 \

python -u run_multivariate.py \
  --batch_size 8 \
  --seq_len 80 \
  --token_t_size 16 \
  --token_t_overlap 0 \
  --token_d_size 100 \
  --token_d_overlap 0 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3 \
  --loss_r 0.75 \

python -u run_multivariate.py \
  --batch_size 8 \
  --seq_len 80 \
  --token_t_size 16 \
  --token_t_overlap 0 \
  --token_d_size 100 \
  --token_d_overlap 0 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3 \
  --loss_r 0.8 \

