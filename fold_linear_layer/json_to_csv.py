import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--region', type=str, help='region eu or us')
parser.add_argument('--fold_index', type=int, help='fold to look at, integers between 0 - 5, 5 indicates the test set')
parser.add_argument('--progress', type=int, help='integer indicator for which progress to look at')
parser.add_argument('--test', action='store_true')
parser.add_argument('--first_model', action='store_true')
parser.add_argument('--model_name', type=str, default='None')

# class Args():
#     def __init__(self):
#         self.progress = 0
#         self.test = False
#         self.model_name = 'loss'
#         self.first_model = True
# args = Args()

if __name__ == '__main__':
    args = parser.parse_args()
    path_dir = f'/nas/guosuhan/gwnet/logs/linears/{args.region}'

    if args.fold_index == 5:
        path_dir = f'/nas/guosuhan/gwnet/logs/linears/{args.region}/test'
    else:
        path_dir = f'/nas/guosuhan/gwnet/logs/linears/{args.region}/fold{args.fold_index}'

    # validation metrics
    path_train = os.path.join(path_dir, 'linear_fold_'+f'{args.progress:03d}', 'meta_data/train_metrics.json')
    data = json.load(open(path_train))

    # data_train = pd.read_json(path_train)
    data_train = pd.DataFrame(data['val_stats'])
    # data_train = data_train.reindex(['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip',  'mape_mask', 'smape' ])
    col1 = pd.Series(['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip',  'mape_mask', 'smape' ])

    values = {}
    for name in col1:
        if 'mape' in name:
            # values[name] = data_train['val_stats'][name]['lowest'] * 100
            values[name] = data_train[name]['lowest'] * 100
        else:
            # values[name] = data_train['val_stats'][name]['lowest']
            values[name] = data_train[name]['lowest']

    out = pd.DataFrame([values])
    out.to_csv('train_metrics.csv')


    if args.test:
        # test metrics
        path_test = os.path.join(path_dir, 'linear_fold_'+f'{args.progress:03d}', 'meta_data/test_metrics.json')
        data = pd.read_json(path_test)
        data = data.reindex(['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip',  'mape_mask', 'smape' ])
        # data.to_csv('test_metrics.csv')
        # data.to_csv('test_metrics.csv', mode='a')

        values={}
        for name in data.index:
            if 'mape' in name:
                values[name] = data['test_stats'][name]
                new_str = name + '_%'
                values[new_str] = data['test_stats'][name] * 100
            else:
                values[name] = data['test_stats'][name]

        out = pd.DataFrame([values])
        out.index = [args.model_name]

        if args.first_model:
            out.to_csv('test_metrics.csv')
        else:
            out.to_csv('test_metrics.csv', mode='a')

