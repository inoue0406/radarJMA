#!/bin/bash

case="result_20200118_persistence"

# running script for Rainfall Prediction with time series prediction model
python ../src/main_TS_persistence.py \
       --test --no_train\
       --test_data_path ../data/data_kanto_ts/_valid_data_2017.csv \
       --test_anno_path ../data/ts_kanto_valid_flatsampled_JMARadar.csv \
       --result_path $case --tdim_use 12 \
       --batch_size 128 \
