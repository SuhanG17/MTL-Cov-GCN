config_dict = {'dataset_path': '../data/data_processing_v2/eu_incidence_rate_no_index.csv', # set to '../data/data_processing_v2/us_incidence_rate_no_index.csv' for US-states
               'target_dim': 28, # set to 28 for EEA-UK, set to 49 for US-states
               'src_seq_len': 12,  # int(past_history_factor * trg_seq_len)
               'trg_seq_len': 3,
               'past_history_factor': 4,
               'norm_method': 'zscore', # support zscore and minmax
               'rand': False, # select validation indices at random
               'ratio': [0.7, 0.2, 0.1], # train, val, test split
               'batch_size': [32, 32, 32],
               'shuffle': [True, True, False],
               'drop_last': [False, False, False],
               'split_by_fold': False, # if split the data by fold
               'current_fold': 5, # currently working on which fold
               'num_folds': 6, # how many fold used
               'num_test_samples': None, #101, # decide how many samples in test set NOT ELEMENT
               'metric_key_list': ['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape'], # ['mae', 'mse', 'rmse', 'mape', 'mape_clip', 'mape_mask','smape'],
               'train_mode': True, # set true for training, only explicitly used in __main__
               'load_net_path': None, # set to none if train from scratch; set to a full path with train_mode=True, if resume training; set to full path with train_mode = False, if testing
               'loss_type': 'mask_mae', 
               'factor': 15.0, # factor in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'warmup': 15000, # warmup steps in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'resume_from_step': int(0*0), # control lr warmup when resume training using format int(num_batches_per_epoch*num_epoch_to_resume_from), if set to 0, meaning training from scratch, if other optimizer is selected, it won't affect anyting 
               'log_lowest_eps': 1e-4, # smallest improvement to renew the loss and metrics for meta_data save
               'root_dir': 'logs/trans', # the directory to save models, runs and meta_data for each train
               'model_name': 'original_transformer_', # model name with underscore
               'process': 1, # process number to run this model
               'lr': 1e-3, # learning rate, only for SGD, Adam and AdamX
               'log_batch': False, # if you want to log metrics and loss for each batch
               'log_hp': True, # if you want to log hyperparams in tunable configs, used for hyperparams search
               'early_stopping': False, # if use early stopping for training
               'tolerance': 5, # number of epochs tolerated before stopping
               'min_delta': 0.5, # the difference between val_loss and train_loss to be considered significant enough to show overfitting
               'save_every': 20, # save model every n round
               'epochs': 100 # total number of epochs to train
            }



tunable_config_dict = {'N': [1, 2, 3], # num stacks for encoder and decoder in original transformer; transformer linear decoder have only one stack
                       'd_ff': [32, 64, 128], # feedforward hiddem dim
                       'd_model': [16, 32, 64], # model hidden dim
                       'dropout': [0.3, 0.4, 0.5], # dropout rate
                       'h': [2, 4, 8], # num of attention heads
                    #    'optimizer_method':['WarmUp'], # optimizer to use
                       'optimizer_method':['AdamX', 'SGD', 'WarmUp'], # optimizer to use
                    }


import os
import json
from itertools import product

class ConfigToClass(object):
  def __init__(self, my_dict):
    for key in my_dict:
        setattr(self, key, my_dict[key])


def configs(hyper_list:list):
    configs = ConfigToClass(config_dict)

    if len(hyper_list) != len(tunable_config_dict.keys()):
        raise ValueError('list should be long enough to select one option for each hyperparam')

    selected_dict = {}
    for i, param in enumerate(tunable_config_dict.items()):
        selected_dict[param[0]] = param[1][hyper_list[i]]

    tunable_config = ConfigToClass(selected_dict)

    return configs, tunable_config

# configs, t_configs = configs([0, 0, 0, 0, 0])

def configs_all():
    configs = ConfigToClass(config_dict)
    selected_dict = dict.fromkeys(tunable_config_dict.keys(),[])

    param_values = [v for v in tunable_config_dict.values()]
    for N, d_ff, d_model, dropout, h, optimizer_method in product(*param_values):
        selected_dict['N'] = N
        selected_dict['d_ff'] = d_ff
        selected_dict['d_model'] = d_model
        selected_dict['dropout'] = dropout
        selected_dict['h'] = h
        selected_dict['optimizer_method'] = optimizer_method 

        yield configs, ConfigToClass(selected_dict)

# temp = configs_all()

# for i, (c, t_c) in enumerate(temp):
#     print(c.__dict__)
#     print(t_c.__dict__)


def save_to_json(input, filename, path):
    path = os.path.join(path, filename)

    if isinstance(input, dict):
        with open(path, 'w', newline='\r\n') as outfile:
            json.dump(input, outfile, indent=4) 
    elif isinstance(input, object):
        with open(path, 'w', newline='\r\n') as outfile:
            json.dump(input.__dict__, outfile, indent=4)  
    else:
        raise ValueError('input should either be class object or dictionary') 

# save_to_json(configs, 'configs.json', '.')
# save_to_json(t_configs, 'tunable_configs.json', '.')
# save_to_json(config_dict, 'configs_dict.json', '.')
