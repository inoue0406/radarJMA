#!/bin/bash

case="result_20190716_clstm_tr_2yr_flat"
# case for transfer learning
case_tr="result_20190715_clstm_alljapan_2yr/trained_CLSTM.model"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_kanto_flatsampled_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --transfer_path $case_tr --tdim_use 12 --loss_function MSE --learning_rate 0.02 --lr_decay 0.7 --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

