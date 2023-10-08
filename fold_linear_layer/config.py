config_dict = {'dataset_path':  ## EU incidence
                                ['../data/data_processing_v2/eu_incidence_rate_no_index.csv'],
                                ## EU incidence + 2 supp
                                # ['../data/data_processing_v2/eu_incidence_rate_no_index.csv',
                                # '../data/data_processing_v2/eu_fatality_rate_no_index.csv',
                                # '../data/data_processing_v2/eu_hospitalization_rate_no_index.csv'],
                                ## US incidence
                                # ['../data/data_processing_v2/us_incidence_rate_no_index.csv'],
                                ## US incidence + 2 supp
                                # ['../data/data_processing_v2/us_incidence_rate_no_index.csv',
                                #  '../data/data_processing_v2/us_fatality_rate_no_index.csv',
                                # '../data/data_processing_v2/us_hospitalization_rate_no_index.csv'],
               'pkl_filename': '../data/data_processing_v2/eu_adj_covid.pkl', # '../data/data_processing_v2/us_adj_covid.pkl'
               'sensors': 28, #49
               'input_dim': 1, #3,
               'target_dim':1, #3
               'src_seq_len': 12, # int(past_history_factor * trg_seq_len)
               'trg_seq_len': 3,
               'past_history_factor': 4, 
               'max_len': 3,
               'pollute': False, # if pollute some sensors with a mask tensor, only work in training
               'naive_vs_geom': 'geom', # which type of pollution to be applied
               'masking_ratio': 0.4, # percent tensors to be polluted, default naive:0.15, geom:0.3
               'all_random': True, # which kind of random matrix, all random or random vector repeat n times
               'gamble': True, # if set 20% of pollute_ratio sensors to gamble
               'lm': 3, # length to pollute for each input
               'mode': 'separate', # choose between concurrent and separate, polluted at same position or different
               'distribution': 'geometric', # which distirbution to use in GeomMLM, can choose random
               'exclude_sensors': None, # list of sensors index to not be polluted
               'exclude_features': [], # features to be excluded from masking
               'norm_method': 'zscore', # support zscore and minmax
               'rand': False, # select validation indices at random
               'ratio': [0.7, 0.2, 0.1], # train, val, test split
               'batch_size': [32, 32, 32], # [128, 128, 128],# [64, 64, 64], 
               'shuffle': [True, True, False],
               'drop_last': [False, False, False],
               'individual': True, # use linear layer by Sensor(channel)
               'split_by_fold': False, # if split the data by fold
               'current_fold': 4, # currently working on which fold, choose between integers within range(0, num_folds-1)
               'num_folds': 6, # how many fold used
               'num_test_samples': 102, #102, # decide how many samples in test set NOT ELEMENT
               'metric_key_list': ['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape'], # ['mae', 'mse', 'rmse', 'mape', 'mape_clip', 'mape_mask','smape'],
               'train_mode': True, # set true for training, only explicitly used in __main__
               'load_net_path': None, # set to none if train from scratch; set to a full path with train_mode=True, if resume training; set to full path with train_mode = False, if testing
               'model_type': 'DLinear', # which model to use, choose among Linear, DLinear, NLinear 
               'loss_type': 'MSE', #'mask_mae', #'MSE', #'mask_mae', 
               'lradj': '5', # adjust learning rate
               'factor': 2.5, #1.4, # factor in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'warmup': 20000, #10000, # warmup steps in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'resume_from_step': int(375*0), #int(188*52), # control lr warmup when resume training, if set to 0, meaning training from scratch, if other optimizer is selected, it won't affect anyting 
               'log_lowest_eps': 1e-3, # smallest improvement to renew the loss and metrics for meta_data save
               'root_dir': '../logs/linears/', # the directory to save models, runs and meta_data for each train
               'model_name': 'linear_fold_', #'gcn_transformer_', # model name with underscore
               'process': 1, # process number to run this model
               'lr': 1e-3, # learning rate, only for SGD, Adam and AdamX
               'log_batch': False, # if you want to log metrics and loss for each batch
               'log_hp': False, # if you want to log hyperparams in tunable configs, used for hyperparams search
               'early_stopping': False, # if use early stopping for training
               'tolerance': 5, # number of epochs tolerated before stopping
               'min_delta': 0.5, # the difference between val_loss and train_loss to be considered significant enough to show overfitting
               'save_every': 30, # save model every n round
               'epochs': 30 # total number of epochs to train
            }

tunable_config_dict = {'optimizer_method':['Adam', 'SGD', 'AdamX']}# optimizer to use



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
    for N, d_ff, d_model, dropout, h, opt in product(*param_values):
        selected_dict['N'] = N
        selected_dict['d_ff'] = d_ff
        selected_dict['d_model'] = d_model
        selected_dict['dropout'] = dropout
        selected_dict['h'] = h
        selected_dict['optimizer_method'] = opt

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