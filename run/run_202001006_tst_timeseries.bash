#!/bin/bash

case="result_202001006_tst_timeseries"

# running script for Rainfall Prediction with time series prediction model
python ../src/main_TS_seq2seq_jma.py --model_name seq2seq\
       --train_data_path ../data/data_kanto_ts/_max_log_2015.csv \
       --train_anno_path ../data/ts_kanto_train_flatsampled_JMARadar.csv \
       --valid_data_path ../data/data_kanto_ts/_max_log_2015.csv \
       --valid_anno_path ../data/ts_kanto_train_flatsampled_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 \
       --test_data_path ../data/data_kanto_ts/_max_log_2015.csv \
       --test_anno_path ../data/ts_kanto_train_flatsampled_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.02 --lr_decay 0.9 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam
