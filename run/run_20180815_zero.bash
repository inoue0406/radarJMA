#!/bin/bash

case="result_20180815_zero"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_persistence_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test_path none --result_path $case --tdim_use 6 --learning_rate 0.01 --batch_size 10 --n_epochs 10 --n_threads 4 --checkpoint 10 --hidden_channels 12 --kernel_size 3