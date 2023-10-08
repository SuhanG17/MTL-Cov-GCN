config_dict = {'dataset_path':  ## EEA-UK incidence
                                ['../data/data_processing_v2/eu_incidence_rate_no_index.csv'],
                                ## EEA-UK incidence + 2 supp
                                # ['../data/data_processing_v2/eu_incidence_rate_no_index.csv',
                                # '../data/data_processing_v2/eu_fatality_rate_no_index.csv',
                                # '../data/data_processing_v2/eu_hospitalization_rate_no_index.csv'],
                                ## US-states incidence
                                # ['../data/data_processing_v2/us_incidence_rate_no_index.csv'],
                                ## US-states incidence + 2 supp
                                # ['../data/data_processing_v2/us_incidence_rate_no_index.csv',
                                #  '../data/data_processing_v2/us_fatality_rate_no_index.csv',
                                # '../data/data_processing_v2/us_hospitalization_rate_no_index.csv'],
               'pkl_filename': '../data/data_processing_v2/eu_adj_covid.pkl', # set to '../data/data_processing_v2/us_adj_covid.pkl' for US-states
               # data pre-processing
               'sensors': 28, # set to 28 for EEA-UK, set to 49 for US-states
               'input_dim': 1, # set to 1 if forecast only incidence; set to 3 if supplementary variables are used, set root_dir below accordingly
               'target_dim':1, # set to 1 if forecast only incidence; set to 3 if supplementary variables are used, set root_dir below accordingly
               'src_seq_len': 12, # int(past_history_factor * trg_seq_len)
               'trg_seq_len': 3,
               'past_history_factor': 4, 
               'max_len': 3, # set to the same value as trg_seq_len 
               'norm_method': 'zscore', # support zscore and minmax
               'rand': False, # select validation indices at random
               'ratio': [0.7, 0.2, 0.1], # train, val, test split
               'batch_size': [8, 8, 8],
               'shuffle': [True, True, False],
               'drop_last': [False, False, False],
               # train on which fold or final test set
               'split_by_fold': False,  # if split the data by fold
               'current_fold': 5, # currently working on which fold
               'num_folds': 6, # how many fold used
               'num_test_samples': None, #101, # decide how many samples in test set NOT ELEMENT
               # parameters related to adjacency map 
               'set_diag': [[False]], # set self-loop to sensory maps, inclusive of both original and adapted
               'undirected': [[False]], # make sensory map symmetric, inclusive of both original and adapted
               'truncate': [[False]], # if truncate sensor maps wrt a threshold, inclusive of both original and adapted 
               'threshold': [[1.]], #  clamp threshold, smaller than threshold will be set to 0.,  inclusive of both original and adapted  
               'sparsity_ratio': [[1.]], # sparsify sensor maps, desired sparsity ratio in each map, inclusive of both original and adapted  
               'num_layers': 2, # k-hop of aggregation neighborhood
               'bn': False, # batch norm for gcn
               'conv_type': ['GCN', 'GCN'], # graph conv type
               'num_maps': [1, 0], # number of maps, original included; if not original map, set to [1, 2]
               'adp_supp_len': [0, 0], # number of adapted maps
               # metrics, train_or_test, optimizer and loss fn 
               'metric_key_list': ['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape'], # ['mae', 'mse', 'rmse', 'mape', 'mape_clip', 'mape_mask','smape'],
               'train_mode': False, # set true for training, only explicitly used in __main__
               'load_net_path': None,# set to none if train from scratch; set to a full path with train_mode=True, if resume training; set to full path with train_mode = False, if testing
               'loss_type': 'MAE', 
               'factor': 15.0, # factor in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'warmup': 15000, # warmup steps in WarmUp optimizer, if other optimizer is selected, it won't affect anyting
               'resume_from_step': int(0*0), # control lr warmup when resume training using format int(num_batches_per_epoch*num_epoch_to_resume_from), if set to 0, meaning training from scratch, if other optimizer is selected, it won't affect anyting 
               'log_lowest_eps': 1e-4, # smallest improvement to renew the loss and metrics for meta_data save
               'lr': 1e-4, # learning rate, only for SGD, Adam and AdamX      
               # early-stopping mechanism 
               'early_stopping': False, # if use early stopping for training
               'tolerance': 5, # number of epochs tolerated before stopping
               'min_delta': 0.5, # the difference between val_loss and train_loss to be considered significant enough to show overfitting
               # save trained model and hyper-parameters
               'root_dir': 'logs/trans_gnn', # logs/trans_gnn_supp # the directory to save models, runs and meta_data for each train
               'model_name': 'gcn_transformer_fold_', # model name with underscore
               'process': 1, # process number to run this model
               'log_batch': False, # if you want to log metrics and loss for each batch
               'log_hp': True, # if you want to log hyperparams in tunable configs, used for hyperparams search
               'save_every': 20, # save model every n epochs
               # Total epochs to train 
               'epochs': 100, # total number of epochs to train
               # +++++++ BEGIN: function not used, but parameters passed, do NOT change ++++++++
               # MLM
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
               # Changing Beta
               'beta_scale': 'stable', # scale used to increase beta, choose between 'linear', 'log' and 'stable'
               'beta_upper_lim': 10, # the maximum of beta values
               'agreement_threshold': 0.3, # agreement expected to stop beta from updating
               'init_value': -1., # init beta to this value, if beta increment not intended set to -1. means to use common softmax
               # +++++++ END: function not used, but parameters passed, do NOT change ++++++++
            }

tunable_config_dict = {'N': [1, 2, 4, 6], # num stacks for encoder and decoder in original transformer; transformer linear decoder have only one stack
                       'd_ff': [64, 128, 256, 512, 1024], # feedforward hiddem dim
                       'd_model': [32, 64, 128, 256, 512], # model hidden dim
                       'dropout': [0.3, 0.4, 0.5], # dropout rate
                       'h': [2, 4, 8], # num of attention heads
                       'optimizer_method':['AdamX', 'SGD', 'WarmUp'], # optimizer to use
                       'add_gcn':[True, False] # add gcn_layer or not
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
