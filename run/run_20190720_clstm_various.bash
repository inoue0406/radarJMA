#!/bin/bash

case="result_20190720_clstm_100sample_lr0002"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.002 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

case="result_20190720_clstm_100sample_bs20"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.02 --batch_size 20 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

case="result_20190720_clstm_100sample_lr00002"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.0002 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

case="result_20190720_clstm_100sample_bs40"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.02 --batch_size 40 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

case="result_20190720_clstm_100sample_ch16"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.02 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 16 --kernel_size 3 --optimizer adam


case="result_20190720_clstm_100sample_ker5"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_100sample_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --loss_function MSE --learning_rate 0.02 --batch_size 10 --n_epochs 40 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 5 --optimizer adam


