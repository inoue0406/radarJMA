#!/bin/bash

# running script for Rainfall Prediction with ConvLSTM

case="run_20180812_lr0001"

python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test_path none --result_path $case --tdim_use 6 --learning_rate 0.001 --batch_size 10 --n_epochs 50 --n_threads 4 --checkpoint 10 --hidden_channels 12 --kernel_size 3

#case="run_20180812_lr00001"

#python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test_path none --result_path $case --tdim_use 6 --learning_rate 0.0001 --batch_size 10 --n_epochs 50 --n_threads 4 --checkpoint 10 --hidden_channels 12 --kernel_size 3

#case="run_20180812_ker5"

#python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test_path none --result_path $case --tdim_use 6 --learning_rate 0.0001 --batch_size 10 --n_epochs 10 --n_threads 4 --checkpoint 10 --hidden_channels 12 --kernel_size 5


