import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from MLM import NaiveMLM, GeomMLM
from sklearn.model_selection import TimeSeriesSplit

# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0

# def make_std_mask(tgt, pad, device):
#     "Create a mask to hide padding and future words."
#     tgt_mask = (tgt != pad).unsqueeze(-2)
#     tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(device)
#     return tgt_mask

def fold_indices(num_folds, num_test_samples, target_len):
    test_size = num_test_samples + target_len - 1 
    tscv = TimeSeriesSplit(n_splits=num_folds, test_size=test_size)
    return tscv


def generate_df_by_fold(data_path, tscv=None):
    df = pd.read_csv(data_path, header=None) # no index, no header
    print(f'df has shape {df.shape}')
    fold_indices = None

    if (tscv) is not None:
        fold_indices = {}
        df_dict = {}
        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            fold_indices[f'fold{i}'] = [train_index, test_index, np.concatenate([train_index, test_index])]
            df_dict[f'fold{i}'] = df.iloc[np.concatenate([train_index, test_index])]
            shape = df_dict[f'fold{i}'].shape
            print(f'fold {i}: df has shape {shape}')
        df = df_dict

    return df, fold_indices

class Scaler(object):
    def __init__(self, df):
        self.df = df
        # print(f'df has shape {df.shape}')

        # self.df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        #            columns=['a', 'b', 'c'])

        self.scaler = {'mean': torch.from_numpy(self.df.mean(axis=0).to_numpy()).unsqueeze(0).float(), 
                       'std': torch.from_numpy(self.df.std(axis=0).to_numpy()).unsqueeze(0).float(), 
                       'max': torch.from_numpy(self.df.max(axis=0).to_numpy()).unsqueeze(0).float(), 
                       'min': torch.from_numpy(self.df.min(axis=0).to_numpy()).unsqueeze(0).float()}


    def transform(self, data, norm_method):
        # data shape: [seq_len, sensors]
        # scaler shape : [1, sensors]
        if norm_method == 'zscore':
            data = (data - self.scaler['mean']) / (self.scaler['std'] + 1e-5)

        elif norm_method == 'minmax':
            data = (data - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'] + 1e-5)

        else:
            raise ValueError('Only zscore and minmax normalization implemented')
        
        return data

    def inverse_transform(self, data, norm_method):
        # data shape: [batch_size, sensors, seq_len, dim] 
        # scaler: [1, sensors] -> [1, sensors, 1, 1]
        if norm_method == 'zscore':        
            data = data * self.scaler['std'].unsqueeze(-1).unsqueeze(-1) + self.scaler['mean'].unsqueeze(-1).unsqueeze(-1)

        elif norm_method == 'minmax':
            data = data * (self.scaler['max'].unsqueeze(-1).unsqueeze(-1) - self.scaler['min'].unsqueeze(-1).unsqueeze(-1)) + self.scaler['min'].unsqueeze(-1).unsqueeze(-1)

        else:
            raise ValueError('Only zscore and minmax normalization implemented')
        
        return data


class SWDataset(Dataset):
    """ Produce data pair in sliding window manner

    refer to 'https://discuss.pytorch.org/t/dataloader-for-a-lstm-model-with-a-sliding-window/22235'

           
    |------------------|
    | sample1 |
     | sample2 |
    
    Attributes:
        scaler list: list of instansitaions of Scaler class, each element is pointing to one variable/file
        norm_method: 'zscore' or 'minmax'
        past_history_factor: input window = int(past_history_factor * forecast horizon)
        target_len: length of target tensor
        target_dim: dim of target tensor, select till target_dim for label tensor

        MLM: instantation of MLM class
        all_random: if create a mask matrix with random numbers entirely, or repeat vector of random numbers n times
        exclude_features: list of var to be excluded from masking
    """
    def __init__(self, scaler, norm_method, past_history_factor, target_len, target_dim, MLM, all_random, exclude_features):
        
        self.scaler = scaler
        self.data =  [torch.from_numpy(var.df.values).float() for var in scaler]
        self.window = int(past_history_factor * target_len)
        self.norm_method = norm_method
        self.target_len = target_len
        self.target_dim = target_dim
        self.start_token = [int(var.scaler['max'].max()) + 1 for var in scaler]
        # make the int(maximum value)+1 the start token

        # MLM strategy
        self.MLM = MLM
        self.all_random = all_random 
        self.exclude_features = exclude_features

    def __getitem__(self, index):
        x_list = []
        y_list = []
        y_raw_list = []
        for i, (scaler, var, token) in enumerate(zip(self.scaler, self.data, self.start_token)):
            x_slice = var[index : index+self.window, :]  # [seq_len, sensors] 
            y_slice =  var[index+self.window : index+self.window+self.target_len, :] # [seq_len, sensors] 
            y_raw_list.append(y_slice)

            _, num_sensors = y_slice.shape
            added_start_token = torch.ones(1, num_sensors).fill_(token)
            y_added =  torch.cat([added_start_token, y_slice], dim=0) # [seq_len + 1, sensors]

            x_norm = scaler.transform(x_slice, self.norm_method)
            y_norm = scaler.transform(y_added, self.norm_method)
            if (self.MLM is not None) and (i not in self.exclude_features): # (MLM has to be provided) and (exclude features list is either empth or does not contain current var)
                x_norm = self.MLM.apply_mask(x_norm, self.all_random)
                    
            x_list.append(x_norm)
            y_list.append(y_norm)

        x = torch.stack(x_list) # [num_vars, seq_len, sensors]
        y = torch.stack(y_list) # [num_vars, seq_len, sensors] 
        y_raw = torch.stack(y_raw_list) # [num_vars, seq_len, sensors]
        # select the first target_dim vars as label
        y, y_raw = y[:self.target_dim, ...], y_raw[:self.target_dim, ...] 

        return x.permute(2, 1, 0), y.permute(2, 1, 0), y_raw.permute(2, 1, 0) 
        # [sensors, seq_len, input_dim], [sensors, seq_len+1, target_dim], [sensors, seq_len, target_dim]

    def __len__(self):
        return self.data[0].shape[0] - (self.window+self.target_len) + 1


def train_val_test_split(dataset, ratio:list, rand=False, num_test_samples=None):
    """ Data Split
    
    Attributes:
        dataset: pytorch object of all data points
        ratio: list of ratio in decimals, [train, val, test], e.g., [0.7, 0.2, 0.1]
        rand: if select validation set indices at random. train and test indices will remain their order.
        num_test_samples: num samples assigned each fold
    Returns:
        `train/val/test`_sets: subset of dataset

    Why subst not sampler?
    Notice that SubsetRandomSampler will return a permutation of indices, and thus not compatible with
    shuffle in dataloader. Hence, the sampler created can not sequential, but shuffled always.
    
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if (num_test_samples is not None):
        test_split = num_test_samples # the test split is assigned
        test_indices = indices[-test_split:]
        trainval_ind = indices[:-test_split]
        val_split = int(np.floor(len(trainval_ind) * ratio[1]))
    else:
        val_split = ratio[1]
        test_split = ratio[-1]

        val_split = int(np.floor(val_split * dataset_size))
        test_split = int(np.floor(test_split * dataset_size))

        trainval_ind = indices[:-test_split]
        test_indices = indices[-test_split:]

    if rand:
        np.random.seed(17)
        val_indices = np.random.choice(len(trainval_ind), val_split, replace=False)
        train_indices = [x for x in trainval_ind if x not in val_indices]         
    else:
        train_indices = trainval_ind[:-val_split]
        val_indices = trainval_ind[-val_split:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print(f'train_set has {len(train_set)} samples, val_set has {len(val_set)} samples, test_set has {len(test_set)} samples')

    return train_set, val_set, test_set 


def dataloader(dataset, batch_size, shuffle=False, drop_last=False):
    """ pytorch dataloader object for datasset

    Args:
        dataset: pytorch dataset object
        batch_size: train/val/test batch_size 

    Returns:
        loader: dataloader object, shape [batch_size, sensors, seq_len, dim]
    """

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        shuffle=shuffle,
                        num_workers=2,
                        pin_memory=True)

    return loader


class Batch:
    """Object for holding a batch of data with mask during training.

    start_token = 1.
    pad = 0.

    src shape: [batch_size, sensors, input_seq_len, input_dim]
    trg shape: [batch_size, sensors, target_seq_len + 1, target_dim] where 1 represents the start_token, should be added in dataset

    if input_dim or target_dim is not one, then the src_mask/trg_mask considers only the mean of dim, works because no padding is applied
    """
    def __init__(self, src, trg=None, pad=0, device='cpu'):
        self.src = src
        # self.src_mask = (src.squeeze(-1) != pad).unsqueeze(-2) #[batch_size, sensors, 1, seq_len]
        self.src_mask = (src.mean(-1) != pad).unsqueeze(-2) #[batch_size, sensors, 1, seq_len]
        if trg is not None:
            self.trg = trg[:, :, :-1, :] # [batch_size, sensors, target_seq_len, target_dim] conatins start_token
            self.trg_y = trg[:, :, 1:, :] # [batch_size, sensors, target_seq_len, target_dim] NO start_token
            # self.trg_mask = make_std_mask(self.trg.squeeze(-1), pad, device) # [batch_size, sensors, target_seq_len, target_seq_len]
            # self.ntokens = (self.trg_y.squeeze(-1) != pad).sum()
            # self.start_token = trg[:, :, 0, :].unsqueeze(-1) # [batch_size, sensors, 1, 1]



def build_dataset(dataset_path, norm_method, past_history_factor, target_len, target_dim, 
                  rand, ratio:list, batch_size:list, shuffle:list, drop_last:list, 
                  fold:int, num_folds:int, num_test_samples:int, **kwargs):
    """ build loader 
    
    all list object should follow order: [train, val, test]

    Notice: Batch should be imported to main.py, used after dataloader iteration 
    """
    # print(f'fold is {fold}')
    if (fold is not None):
        tscv = fold_indices(num_folds, num_test_samples, target_len)
        df_list = []
        for dset in dataset_path:
            df, indices = generate_df_by_fold(dset, tscv)
            df_list.append(df)
        # save indicies
        # torch.save(indices, f'fold_split_indices_{num_folds}folds.pt')
        if (kwargs['save_tensor'] is not None):
            kwargs['save_tensor'](indices, f'fold_split_indices_{num_folds}folds.pt')

        scaler_list = []
        for df in df_list:
            scaler_list.append(Scaler(df[f'fold{fold}']))
    
        print(f'=============================fold{fold}=============================') 
    else:
        scaler_list = []
        for dset in dataset_path:
            df = pd.read_csv(dset, header=None) # no index, no header 
            scaler_list.append(Scaler(df))


    if kwargs['pollute']:
        if 'gamble' in kwargs.keys():
            mlm = NaiveMLM(kwargs['seq_len'], kwargs['sensors'], kwargs['masking_ratio'], kwargs['save_tensor'], kwargs['gamble'])
            print('mask with naive MLM')
        else:
            mlm = GeomMLM(kwargs['seq_len'], kwargs['sensors'], kwargs['masking_ratio'], kwargs['save_tensor'], 
                          kwargs['lm'], kwargs['mode'], kwargs['distribution'], kwargs['exclude_sensors'])
            print('mask with geom MLM')
        all_random = kwargs['all_random']
        exclude_features = kwargs['exclude_features']
    else:
        mlm = None
        all_random = None
        exclude_features = []
        print('no mask applied')
    
    dataset = SWDataset(scaler_list, norm_method, past_history_factor, target_len, target_dim, mlm, all_random, exclude_features)

    train_set, val_set, test_set = train_val_test_split(dataset, ratio, rand, num_test_samples)
    train_loader = dataloader(train_set, batch_size[0], shuffle[0], drop_last[0])
    val_loader = dataloader(val_set, batch_size[1], shuffle[1], drop_last[1])
    test_loader = dataloader(test_set, batch_size[2], shuffle[2], drop_last[2])

    return train_loader, val_loader, test_loader, scaler_list



# from config import configs
# config, t_config = configs([0])
# tensors_save = None


# config.num_test_samples = 102
# config.split_by_fold = True
# fold = config.current_fold
# train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
#                                                                 config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
#                                                                 fold=fold, num_folds=config.num_folds, num_test_samples=config.num_test_samples, 
#                                                                 pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, lm=config.lm, mode=config.mode, distribution=config.distribution, exclude_sensors=config.exclude_sensors, all_random=config.all_random, exclude_features=config.exclude_features)

# config.split_by_fold = False
# config.num_test_samples = None
# train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
#                                                                 config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
#                                                                 fold=None, num_folds=config.num_folds, num_test_samples=config.num_test_samples,
#                                                                 pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, lm=config.lm, mode=config.mode, distribution=config.distribution, exclude_sensors=config.exclude_sensors, all_random=config.all_random, exclude_features=config.exclude_features)

