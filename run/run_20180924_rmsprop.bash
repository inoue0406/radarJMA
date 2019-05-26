#!/bin/bash

case="result_20180917_adam"

# with learning rate decay

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --data_scaling linear --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --optimizer adam --learning_rate 0.02 --lr_decay 0.9 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 >& log-$case

case="result_20180917_rmsprop"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_h5/ --data_scaling linear --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --optimizer rmsprop --learning_rate 0.02 --lr_decay 0.9 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3  >& log-$case
