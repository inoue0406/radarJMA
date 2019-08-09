#!/bin/bash

case="result_20190809_clstm_memorycheck_ep20"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --data_scaling linear --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --model_name clstm_skip --tdim_use 12 --loss_function MultiMSE --loss_weights 1.0 0.05 0.0125 0.0025 --learning_rate 0.02 --lr_decay 0.9 --batch_size 6 --n_epochs 20 --n_threads 4 --checkpoint 10 --hidden_channels 64 --kernel_size 3 --optimizer adam

# post plotting
python ../src_post/plot_comp_prediction.py $case
# post animation generation
python ../src_post/gif_animation.py $case

