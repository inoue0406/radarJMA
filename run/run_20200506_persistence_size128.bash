#!/bin/bash

case="result_20200506_persistence_size128"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_persistence_jma.py --data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv \
       --test --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.01 \
       --batch_size 100 --n_epochs 10 --n_threads 4 --checkpoint 10 \
       --hidden_channels 12 --kernel_size 3 --eval_threshold 0.5 10 20
