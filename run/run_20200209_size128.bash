#!/bin/bash

case="result_20200209_size200"

# running script for Rainfall Prediction with time series prediction model
python ../src/main_TS_tsconv_jma.py --model_name seq2seq\
       --train_data_path ../data/data_kanto_ts/_train_data_2015-2016.csv \
       --train_anno_path ../data/ts_kanto_train_flatsampled_JMARadar.csv \
       --valid_data_path ../data/data_kanto_ts/_valid_data_2017.csv \
       --valid_anno_path ../data/ts_kanto_valid_flatsampled_JMARadar.csv \
       --test \
       --use_var rmax_100 rmean_100 --data_resize 200\
       --test_data_path ../data/data_kanto_ts/_valid_data_2017.csv \
       --test_anno_path ../data/ts_kanto_valid_flatsampled_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.002 --lr_decay 1.0 \
       --batch_size 6 --n_epochs 30 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam


