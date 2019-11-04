#!/bin/bash

# based on result_20190625_clstm_lrdecay07_ep20, the size of image file is reduced as 200->128

case="result_20191103_clstm_size128"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto_resize/ \
       --valid_data_path ../data/data_kanto_resize/ \
       --train_path ../data/train_simple_JMARadar.csv \
       --valid_path ../data/train_simple_JMARadar.csv \
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.02 --lr_decay 0.7 \
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --hidden_channels 8 --kernel_size 3 --optimizer adam
