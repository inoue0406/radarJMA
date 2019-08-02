#!/bin/bash

case="result_20190801_clstm_skip_maxmse3_ep20"

# running script for Rainfall Prediction with ConvLSTM
#python ../src/main_CLSTM_EP_jma.py --data_path ../data/data_kanto/ --valid_data_path ../data/data_kanto/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --data_scaling linear --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --model_name clstm_skip --tdim_use 12 --loss_function MaxMSE --loss_weights 1.0 0.01 --learning_rate 0.02 --lr_decay 0.7 --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam

# post plotting
#python ../src_post/plot_comp_prediction.py $case
# post animation generation
python ../src_post/gif_animation.py $case
