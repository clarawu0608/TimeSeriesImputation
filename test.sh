#!/bin/bash
# Activate virtual environment if needed
. venv/bin/activate

# Run the script with arguments
python -u run.py \
  --batch_size 16 \
  --seq_len 128 \
  --token_size 16 \
  --token_overlap 4 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.3

python -u run.py \
  --batch_size 16 \
  --seq_len 128 \
  --token_size 16 \
  --token_overlap 4 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.4

python -u run.py \
  --batch_size 16 \
  --seq_len 128 \
  --token_size 16 \
  --token_overlap 4 \
  --epoch_num 250 \
  --missing_type 1 \
  --lm 15 \
  --missing_rate 0.5
