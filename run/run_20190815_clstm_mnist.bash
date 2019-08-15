#!/bin/bash

case="result_20190815_mnist_clstm_ep20"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_CLSTM_EP_mnist.py --data_scaling linear --test --eval_threshold 0.0 --result_path $case --model_name clstm_skip --tdim_use 10 --loss_function MSE --learning_rate 0.02 --lr_decay 0.9 --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 --hidden_channels 32 --kernel_size 3 --optimizer adam

# post plotting
#python ../src_post/plot_comp_prediction.py $case
# post animation generation
#python ../src_post/gif_animation.py $case

