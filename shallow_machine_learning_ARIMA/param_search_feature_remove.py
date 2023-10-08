""" Important Suggestion
BE CAREFUL if use gpu with xgboost and lightgbm. For some reason, it creates grid_search instances on GPU device ALL as once, which will cause "Out of Memory" Error.
More importantly, the processes will not automatically be terminated. They have to be killed by hand, which is a loathsome task.
As a result, cpu is a better and somehow, faster choice for grid search.
"""

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import itertools
import sys
sys.stdout.flush()

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from feature_importance import data_split, report_importance, correlation_heatmap
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.arima import AutoARIMA
import numpy as np
import torch
import pickle
from metrics import build_metric
import argparse

import time
import datetime 

parser = argparse.ArgumentParser(
                    prog = 'param_search',
                    description = 'search for best shallow learining params by fold',
                    epilog = 'each fold is searched and tested for result')
parser.add_argument('--model_type', type=str, default='xgboost', help='type of model to use, xgboost, lightgbm and randomforest') 
parser.add_argument('--fold', type=int, default=5, help='number of folds split, used two times')
parser.add_argument('--num_test_samples', type=int, default=102, help='number of test samples in NN')
parser.add_argument('--input_len', type=int, default=12, help='input seq len')
parser.add_argument('--target_len', type=int, default=3, help='target seq len')
parser.add_argument('--dir_path', type=str, default='../data/data_processing_v2', help='dir for all var spreadsheets')
parser.add_argument('--region', type=str, default='eu')
parser.add_argument('--data_file', type=str, default='eu_incidence_rate_with_index.csv', help='path to data csv to be used')
parser.add_argument('--data_type', type=str, default='incidence_eu', help='for loc_wise, indicates which data type is in use; for supp_wise, which target_var to use')
parser.add_argument('--search_type', type=str, default='loc_wise', help='search for what type, loc_wise, supp_wise, arima')
parser.add_argument('--num_supp_vars', type=int, default=1, help='number of supp vars: 1: [incidence] \
                                                                                       3: [incidence, fatality, hospitalization] \
                                                                                       4: [incidence, fatality, hospitalization, vaccination] \
                                                                                       5: [incidence, fatality, hospitalization, vaccination, testing] \
                                                                                       6: [incidence, fatality, hospitalization, vaccination, testing, traffic]')
# parser.add_argument('--target_var', type=str, default='incidence', help='supp_wise search, predict which variable, much be in var_dict')
# parser.add_argument('--var_dict', type=dict, default={}, help='vars used in supp wise search')

# class Args():
#     def __init__(self) -> None:
#         self.model_type = 'xgboost'
#         self.fold = 5
#         self.num_test_samples = 102
#         self.input_len = 12
#         self.target_len = 3
#         self.dir_path = '/data/guosuhan_new/st_gnn/graph_transformer/covid/covid_no_sensors/data_processing_v2' 
#         self.region = 'eu'
#         self.data_file = 'eu_incidence_rate_with_index.csv'
#         self.data_type = 'incidence_eu'
#         self.search_type = 'loc_wise'
#         self.num_supp_vars = 3
# args = Args()

def univariate_autoarima(train_data, target_len, max_p, max_q):
    '''Use autoARIMA function to search for best arima params
    Args:
        train_data: pandas series which should be longer than expected lags
        target_len: same as in nn
        max_p: maximum AR lag, should be equal to input_seq_len
        max_q: maximum MA window size, should be equal to input_seq_len
        refer to http://alkaline-ml.com/pmdarima/tips_and_tricks.html#understand-p-d-and-q

    Returns:
        y_pred: predicted series
        forecaster: the model
    '''

    forecaster = AutoARIMA(sp=7, d=None, max_p=max_p, max_q=max_q, suppress_warnings=True)  
    # forecaster = AutoARIMA(d=None, max_p=max_p, max_q=max_q, seasonal=False, suppress_warnings=True)  

    # sp stands for daily data, only use for seasonal data
    # d is searched using kpss unit root text
    forecaster.fit(train_data) 

    predict_list = [*range(1, target_len+1, 1)]
    y_pred = forecaster.predict(fh=predict_list)

    return y_pred, forecaster


def form_dataframe_by_sensor(sensor_name, dir_path, var_file_path:dict):
    ''' form csv spreadsheet from supplementary vars for each loc '''
    index_names = pd.read_csv(os.path.join(dir_path, var_file_path['incidence']), index_col=0, header=0).index

    data_slices = {}
    for key, filename in var_file_path.items():
        df = pd.read_csv(os.path.join(dir_path, filename), index_col=0, header=0)
        data_slice = df[sensor_name]
        data_slice.index = index_names 
        data_slices[key] = data_slice

    new_df = pd.concat(data_slices, axis=1)
    new_df.index.name = None # no need to set name for index

    return new_df

def get_model(model_name, max_depth, n_estimators, lr_or_max_features):
    if model_name == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 learning_rate=lr_or_max_features,
                                 nthread=4,
                                 seed=42,
                                 gpu_id=-1, 
                                 tree_method='gpu_hist')
    elif model_name == 'randomforest':
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      max_features=lr_or_max_features,
                                      random_state=42)
    elif model_name == 'lightgbm':
        model = lgb.LGBMRegressor(num_leaves=31,
                                  max_depth=max_depth,
                                  n_estimators=n_estimators,
                                  learning_rate=lr_or_max_features,
                                  random_state=42)
                                #   device='gpu') # gpu not compatible with cuda
    return model

def timeseries_fold_indices(fold, test_size, train_data):
    tscv = TimeSeriesSplit(n_splits=fold, test_size=test_size)

    fold_indices = {}
    for i, (train_index, test_index) in enumerate(tscv.split(train_data)):
        fold_indices[i] = [train_index, test_index]

    return fold_indices

def perform_search_by_fold(X_train, X_test, y_train, y_test, 
                           fold, fold_indices, target_len,
                           model_name, params_comb, 
                           metric_key_list=[], save_pred=None):        
    yhats = {}
    metrics_dict = {}
    for i in range(fold):
        fold_X_train = X_train.iloc[fold_indices[i][0]]
        fold_y_train = y_train.iloc[fold_indices[i][0]]
        fold_X_test = X_train.iloc[fold_indices[i][1]]
        fold_y_test = y_train.iloc[fold_indices[i][1]]

        current_best = [0, np.inf] # [fold index, best mae]
        for md, ne, lr_or_mf in params_comb:
            model = get_model(model_name, max_depth=md, n_estimators=ne, lr_or_max_features=lr_or_mf)     
            model.fit(fold_X_train, fold_y_train.values.ravel())
            yhat = model.predict(fold_X_test)
            ytrue = fold_y_test.to_numpy().squeeze()
            metrics = build_metric(torch.from_numpy(yhat), torch.from_numpy(ytrue), metric_key_list=metric_key_list)

            if metrics['mae'] < current_best[1]:
                current_best[0] = fold
                current_best[1] = metrics['mae']
                yhats[f'fold{i}'] = yhat
                metrics_dict[f'fold{i}'] = metrics

        print('Fold {}: \n {}'.format(i, metrics_dict[f'fold{i}']))

    current_best_test = ['test', np.inf] # [fold index, best mae]
    for md, ne, lr_or_mf in params_comb:
        test_model = get_model(model_name, max_depth=md, n_estimators=ne, lr_or_max_features=lr_or_mf)
        test_model.fit(X_train, y_train.values.ravel())
        test_yhat = test_model.predict(X_test)
        test_metrics = build_metric(torch.from_numpy(test_yhat), torch.from_numpy(y_test.to_numpy().squeeze()), metric_key_list=metric_key_list) 

        if metrics['mae'] < current_best_test[1]:
            current_best_test[1] = metrics['mae']
            yhats['test'] = test_yhat
            metrics_dict['test'] = test_metrics 
            best_model = test_model
        
    print('Test set: \n {}'.format(metrics_dict['test']))

    if (save_pred is not None):
        torch.save(yhats, os.path.join(save_pred, f'outputs_{model_name}_{target_len}.pt'))
        metrics_df = pd.DataFrame.from_dict(metrics_dict).transpose()
        metrics_df.to_csv(os.path.join(save_pred, f'metrics_{model_name}_{target_len}.csv'))
    
        # Save the model under the cwd
        pkl_filename = os.path.join(save_pred, f'best_{model_name}_{target_len}.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(best_model, file)


def arima_fold_search(fold, fold_indices, input_len, target_len, X_train, y_test, test_size, metric_key_list=[], save_pred=None):
    yhats = {}
    metrics_dict = {}
    for i in range(fold):
        # search for param
        X = X_train[fold_indices[i][0]]
        ytrue = X_train[fold_indices[i][1]]
        yhat, best_model = univariate_autoarima(train_data=X, target_len=test_size, max_p=input_len, max_q=input_len)

        # test result
        metrics = build_metric(torch.from_numpy(yhat), torch.from_numpy(ytrue), metric_key_list=metric_key_list)
        # print(f'y_test: {yhat}')
        # print(f'y_true: {ytrue}')
        print(f'Fold {i}: \n {metrics}')
        yhats[f'fold{i}'] = yhat
        metrics_dict[f'fold{i}'] = metrics

    test_yhat, best_model = univariate_autoarima(train_data=X_train, target_len=test_size, max_p=input_len, max_q=input_len)
    # print(f'test pred {test_yhat}')
    test_metrics = build_metric(torch.from_numpy(test_yhat), torch.from_numpy(y_test), metric_key_list=metric_key_list) 
    print(f'Test set: \n {test_metrics}')
    yhats['test'] = test_yhat
    metrics_dict['test'] = test_metrics

    if (save_pred is not None):
        torch.save(yhats, os.path.join(save_pred, f'outputs_arima_{target_len}.pt'))
        metrics_df = pd.DataFrame.from_dict(metrics_dict).transpose()
        metrics_df.to_csv(os.path.join(save_pred, f'metrics_arima_{target_len}.csv'))
    
        # Save the model under the cwd
        pkl_filename = os.path.join(save_pred, f'best_arima_{target_len}.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(best_model, file)

def main(args):
    test_size = args.num_test_samples + args.target_len - 1 
    print(f'{test_size} elements will be used')

    path = os.path.join(args.dir_path, args.data_file)
    data = pd.read_csv(path, index_col=0, header=0)

    for loc in data.columns:
        if len(loc.split(' ')) > 1: # space exists 
            save_loc = '_'.join(loc.split(' '))
        else:
            save_loc = loc
        # data_type = args.data_type #'incidence_eu'
        # loc = 'Germany'
        if args.search_type =='loc_wise':
            path_to_save = f'../logs/shallow_ml_v2/locationwise/{args.data_type}_{save_loc}/'
        elif args.search_type =='arima':
            path_to_save = f'../logs/shallow_ml_v2/arima/{args.data_type}_{save_loc}/' 
        elif args.search_type == 'supp_wise':
            supp_name = [supp for supp in var_dict.keys() if supp != 'incidence']
            supp_name = '_'.join(supp_name)
            path_to_save = f'../logs/shallow_ml_v2/suppwise/{args.data_type}_{save_loc}_{supp_name}/' 

        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save, exist_ok=True)
        
        if args.search_type =='loc_wise':
            X_train, X_test, y_train, y_test = data_split(data, col_name=loc, test_size=test_size, shuffle=False)
        elif args.search_type =='arima':
            data_pooled = data[[loc]]
            X_train = data_pooled[:-test_size].to_numpy().squeeze()
            y_test = data_pooled[-test_size:].to_numpy().squeeze()
        elif args.search_type == 'supp_wise':
            data_pooled = form_dataframe_by_sensor(sensor_name=loc, dir_path=args.dir_path, var_file_path=var_dict)
            target_var = args.data_type.split('_')[0]
            X_train, X_test, y_train, y_test = data_split(data_pooled, col_name=target_var, test_size=test_size, shuffle=False)

        fold_indices = timeseries_fold_indices(args.fold, test_size, X_train)

        if args.search_type == 'arima':
            print(f'==================Start Searching: {args.search_type}==================')
            print(f'==================loc: {loc}==================')
            arima_fold_search(args.fold, fold_indices, args.input_len, args.target_len, X_train, y_test, test_size,
            ['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape'], path_to_save)
            print('==================Finished!==================')
        else:
            if args.model_type == 'xgboost':
                print(f'using xgboost as the model')
                grid_params = {'max_depth': [int(x) for x in np.linspace(5, 55, 3)], # 3
                               'n_estimators': list(range(100, 1000, 400)), # 3
                               'learning_rate':[0.1, 0.01, 0.05]} # 3
                param_list = [grid_params['max_depth'], grid_params['n_estimators'], grid_params['learning_rate']]
                param_comb = list(itertools.product(*param_list))

            elif args.model_type == 'randomforest':
                grid_params = {'n_estimators': list(range(100, 1000, 400)), # 3
                               'max_depth': [int(x) for x in np.linspace(5, 55, 3)], # 3
                               'max_features': [1.0, 'sqrt', 'log2']} # changed 'auto' to 1.0 # 3
                param_list = [grid_params['max_depth'], grid_params['n_estimators'], grid_params['max_features']]
                param_comb = list(itertools.product(*param_list))
                
            elif args.model_type == 'lightgbm':
                grid_params= {'max_depth':  [int(x) for x in np.linspace(5, 55, 3)], # 3
                              'n_estimators': list(range(100, 1000, 400)), # 3
                              'learning_rate': [0.1, 0.01, 0.05]}# 3
                param_list = [grid_params['max_depth'], grid_params['n_estimators'], grid_params['learning_rate']]
                param_comb = list(itertools.product(*param_list))
            else:
                raise NotImplementedError 

            print(f'==================Start Searching: {args.search_type}==================')
            print(f'==================loc: {loc}==================')
            s1 = time.time()
            perform_search_by_fold(X_train, X_test, y_train, y_test,
                                   args.fold, fold_indices, args.target_len, args.model_type, param_comb,
                                   ['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape'], path_to_save)
            s2 = time.time()
            print(f'==================Finished! Total time {s2-s1} =================')
    print('Done!')    
        

if __name__ == '__main__':
    args = parser.parse_args()
    # print(args)

    var_dict = {'incidence':f'{args.region}_incidence_rate_with_index.csv', 
                'fatality':f'{args.region}_fatality_rate_with_index.csv', 
                'hospitalization': f'{args.region}_hospitalization_rate_with_index.csv', 
                'vaccination': f'{args.region}_vaccination_rate_with_index.csv', 
                'testing': f'{args.region}_testing_with_index.csv',
                'traffic': f'{args.region}_airtraffic_increase_rate_with_index.csv'} 
    
    var_dict = dict(list(var_dict.items())[:args.num_supp_vars])

    for key, value in var_dict.items():
        assert os.path.exists(os.path.join(args.dir_path, value)), f'path: {value}, does not exists!'

    # for loop for shallow_ml
    # set for model type
    # args.model_type = 'xgboost'

    # for target_len in [3, 6, 12]:
    #     args.input_len = 12
    #     args.target_len = target_len

    #     if args.target_len == 12:
    #         args.num_test_samples = 101
    #     else:
    #         args.num_test_samples = 102

    # for target_len in [2, 7, 14]:
    #     args.input_len = 14
    #     args.target_len = target_len

    #     if args.target_len == 14:
    #         args.num_test_samples = 101
    #     else:
    #         args.num_test_samples = 102
        
    #     print(f'model type: {args.model_type}, target_len: {args.target_len}, num_test_samples: {args.num_test_samples}')

    #     print(args)

    #     training_time_0 = time.time()

    #     main(args)

    #     training_time = time.time() - training_time_0
    #     # print(training_time) # seconds
    #     print(str(datetime.timedelta(seconds=training_time))) # days, hours:minutes:seconds
 
    # all looped over
    for model_type in ['lightgbm', 'randomforest', 'xgboost']: #['xgboost', 'lightgbm', 'randomforest']:
        args.model_type = model_type

        for target_len in [3, 6, 12]:
            args.input_len = 12
            args.target_len = target_len

            if args.target_len == 12:
                args.num_test_samples = 101
            else:
                args.num_test_samples = 102

        # for target_len in [2, 7, 14]:
        #     args.input_len = 14
        #     args.target_len = target_len

        #     if args.target_len == 14:
        #         args.num_test_samples = 101
        #     else:
        #         args.num_test_samples = 102
            
            print(f'model type: {args.model_type}, target_len: {args.target_len}, num_test_samples: {args.num_test_samples}')

            print(args)
            main(args)
    
    for model_type in ['lightgbm', 'randomforest', 'xgboost']: #['xgboost', 'lightgbm', 'randomforest']:
        args.model_type = model_type

        # for target_len in [3, 6, 12]:
        #     args.input_len = 12
        #     args.target_len = target_len

        #     if args.target_len == 12:
        #         args.num_test_samples = 101
        #     else:
        #         args.num_test_samples = 102

        for target_len in [2, 7, 14]:
            args.input_len = 14
            args.target_len = target_len

            if args.target_len == 14:
                args.num_test_samples = 101
            else:
                args.num_test_samples = 102
            
            print(f'model type: {args.model_type}, target_len: {args.target_len}, num_test_samples: {args.num_test_samples}')

            print(args)
            main(args)
 

    # for loop for arima
    # args.search_type = 'arima'
    # for target_len in [3, 6, 12]:
    #     args.input_len = 12
    #     args.target_len = target_len

    #     if args.target_len == 12:
    #         args.num_test_samples = 101
    #     else:
    #         args.num_test_samples = 102

    #     print(args)
    #     main(args)

    # for target_len in [2, 7, 14]:
    #     args.input_len = 14
    #     args.target_len = target_len

    #     if args.target_len == 14:
    #         args.num_test_samples = 101
    #     else:
    #         args.num_test_samples = 102
        
    #     print(f'search type: {args.search_type}, target_len: {args.target_len}, num_test_samples: {args.num_test_samples}')
        
    #     print(args)
    #     main(args)

    # single
    # main(args)




