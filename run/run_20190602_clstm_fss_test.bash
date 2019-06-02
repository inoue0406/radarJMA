#!/bin/bash

case="result_20190602_clstm_fss"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --no_train --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --learning_rate 0.002 --batch_size 10 --n_epochs 30 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3
