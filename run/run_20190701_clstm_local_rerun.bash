#!/bin/bash

case="result_20190701_clstm_local_rerun_ep20"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_MUL_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --learning_rate 0.002 --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

