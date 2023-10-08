#!/bin/sh

# Train
# Set train_mode to 'True'
# Set load_net_path to 'None'
CUDA_VISIBLE_DEVICES=0,1 python trainer.py > training_output.log 2>&1 & echo $! > command.pid

# Test
# Set train_mode to 'False'
# Set load_net_path to where model is saved, e.g. '../logs/trans_adp/gcn_transformer_fold_001/model/best_loss_model_.pth'
# CUDA_VISIBLE_DEVICES=0,1 python trainer.py > testing_output.log 2>&1 & echo $! > command.pid

# command explantion
# CUDA_VISIBLE_DEVICES=0,1 ... 2>&1 & : if multiple gpu exists, choose NO.0 and 1 to run
# > training_output.log : save training log to eyeball training and validation metrics
# echo $! > command.pid : save pid number if current run should be interrupted 




