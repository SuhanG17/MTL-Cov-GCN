import pandas as pd
import os
# import json
# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--progress', type=int, help='integer indicator for which progress to look at')
# parser.add_argument('--test', action='store_true')
# parser.add_argument('--first_model', action='store_true')
# parser.add_argument('--model_name', type=str, default='None')

def main(args):
    # test metrics
    path_test = os.path.join(path_dir, 'linear_fold_'+f'{args.process:03d}', 'meta_data/test_metrics.json')
    data = pd.read_json(path_test)
    data = data.reindex(['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip',  'mape_mask', 'smape' ])
    # data.to_csv('test_metrics.csv')
    # data.to_csv('test_metrics.csv', mode='a')

    values={}
    values_mask={}
    for name in data.index:
        if name in ['mae', 'rmse']:
            values[name] = data['test_stats'][name] 
        elif name in ['mae_mask', 'rmse_mask']:
                values_mask[name] = data['test_stats'][name]
        elif name == 'mape_clip':
            new_str = name + '_%'
            values[new_str] = data['test_stats'][name] * 100
            values_mask[new_str] = data['test_stats'][name] * 100


    out, out_mask = pd.DataFrame([values]), pd.DataFrame([values_mask])
    out.index = [f'{args.target_len}_{args.num_supp_var}']
    out_mask.index = [f'mask_{args.target_len}_{args.num_supp_var}'] 

    return out, out_mask

# def output_all(output_list):
#     ones = pd.concat([output_list[0].reset_index(drop=True), output_list[3].reset_index(drop=True), output_list[6].reset_index(drop=True)], axis=1) 
#     fours = pd.concat([output_list[1].reset_index(drop=True), output_list[4].reset_index(drop=True), output_list[7].reset_index(drop=True)], axis=1) 
#     fives = pd.concat([output_list[2].reset_index(drop=True), output_list[5].reset_index(drop=True), output_list[8].reset_index(drop=True)], axis=1) 
#     ones.index = ['one']
#     fours.index = ['four']
#     fives.index = ['five']

#     return pd.concat([ones, fours, fives])

def output_all(output_list):
    ones = pd.concat([output_list[0].reset_index(drop=True), output_list[2].reset_index(drop=True), output_list[4].reset_index(drop=True)], axis=1) 
    fours = pd.concat([output_list[1].reset_index(drop=True), output_list[3].reset_index(drop=True), output_list[5].reset_index(drop=True)], axis=1) 
    ones.index = ['one']
    fours.index = ['four']

    return pd.concat([ones, fours])


if __name__ == '__main__':
    # args = parser.parse_args()
    class Args():
        def __init__(self):
            self.process = None
            self.num_supp_var = 0
            self.target_len = 3
            self.first_model = True
            self.option = 3
            self.fold = 'test' #fold1


    args = Args()

    # path_dir = f'/nas/guosuhan/gwnet/logs/linears/eu/{args.fold}'
    # option_list = [[1, 8, 15], [4, 11, 18], [22, 29, 36], [25, 32, 39]]
    # only incidence, trg_len 3, 6, 12
    # with fat * hos, trg_len 3, 6, 12
    # only incidence, trg_len 2, 7, 14
    # with fat * hos, trg_len 2, 7, 14

    path_dir = f'/nas/guosuhan/gwnet/logs/linears/us/{args.fold}'
    option_list = [[1, 7, 13], [4, 10, 16], [19, 25, 31], [22, 28, 34]]


    outs = []
    outs_mask = []
    for i in option_list[args.option]:
        args.process = i
        out, out_mask = main(args)
        outs.append(out)
        outs_mask.append(out_mask)

    outs_df = pd.concat([outs[0].reset_index(drop=True), 
                        outs[1].reset_index(drop=True), 
                        outs[2].reset_index(drop=True)], axis=1) 
    
    outs_mask_df = pd.concat([outs_mask[0].reset_index(drop=True), 
                                outs_mask[1].reset_index(drop=True), 
                                outs_mask[2].reset_index(drop=True)], axis=1) 
    
    outs_df.to_csv('all.csv')
    outs_mask_df.to_csv('all.csv', mode='a')



    # out_ls = []
    # out_mask_ls = []
    # for args.process in range(191, 197):
    #     if args.process%2 == 1:
    #         args.num_supp_var = 1
    #     elif args.process%2 == 0:
    #         args.num_supp_var = 4  

    #     # if args.process%3 == 2:
    #     #     args.num_supp_var = 1
    #     # elif args.process%3 == 0:
    #     #     args.num_supp_var = 4  
    #     # elif args.process%3 == 1:
    #     #     args.num_supp_var = 5

    #     if args.process < 47:
    #         args.target_len = 3
    #     elif args.process >= 47 and args.process < 50:
    #         args.target_len = 6
    #     elif args.process >= 50 and args.process < 53:
    #         args.target_len = 12

    #     print(f'process: {args.process}, num_supp_var: {args.num_supp_var}, target_len: {args.target_len}') 

    #     out, out_mask = main(args)

    #     out_ls.append(out)
    #     out_mask_ls.append(out_mask)

    # aa = output_all(out_ls)
    # bb = output_all(out_mask_ls)

    # aa.to_csv('all.csv')
    # bb.to_csv('all.csv', mode='a')

