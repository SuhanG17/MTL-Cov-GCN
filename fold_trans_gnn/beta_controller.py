import numpy as np
import torch
from math import e
import os

def beta_range(beta_scale, beta_upper_lim, num_samples, num_epochs):
    ''' beta range, enable linear or log growth'''
    if beta_scale == 'linear':
        return np.linspace(0., beta_upper_lim, num_epochs*num_samples)
    elif beta_scale == 'log':
        # return np.logspace(-9, np.log(beta_upper_lim), base=e, num=num_epochs*num_samples) # starts from 0
        return np.logspace(0, np.log(beta_upper_lim), base=e, num=num_epochs*num_samples) # starts from 1
    elif beta_scale == 'stable':
        return np.ones(num_epochs*num_samples)
    else:
        raise NotImplementedError

def init_beta(beta_nums, N, num_gpus, init_value=0, device='cpu'):
    ''' init beta for networks
    Args:
        beta_nums: number of beta needed for the network, default 3, 1 for encoder, 2 for decoder, 
        N: stack number in transformer
        num_gpus: number of gpus, replicate beta for data.parallel
        init_value: float, init beta with this value
        device: gpu or cpu
    Returns:
        beta: shape [num_gpu(s), N, beta_nums]
    '''
    beta = torch.ones(beta_nums).unsqueeze(0).repeat(N, 1)
    beta = beta.unsqueeze(0).repeat(num_gpus, 1, 1)
    beta = beta.fill_(init_value)
    return beta.to(device)

def init_updated_index(beta_nums, N, device='cpu'):
    ''' init updated index for networks, no need for dataparallel
    Args:
        beta_nums: number of beta needed for the network, default 3, 1 for encoder, 2 for decoder, 
        N: stack number in transformer
        device: gpu or cpu
    Returns:
        updated_index: shape [N, beta_nums]
    '''
    updated_index = torch.ones(beta_nums, dtype=int).unsqueeze(0).repeat(N, 1) 
    return updated_index.to(device)

def beta_load(load_net_path, filetype:str):
    ''' load beta or updated index based on model selected
    Args:
        load_net_path: model selected
        filetype: beta or updated_index to be loaded, w/o .pt 
    Returns:
        tensor_loaded: loaded beta or updated_index
    '''

    path = '/'.join(load_net_path.split('/')[:-1])
    filename = load_net_path.split('/')[-1] 
    if 'best' in filename:
        tensor_ind = filename.split('_')
        for i in range(len(tensor_ind)):
            if tensor_ind[i] == 'model':
                tensor_ind[i] = filetype
            if tensor_ind[i] == '.pth':
                tensor_ind[i] = '.pt'
        tensor_name = '_'.join(tensor_ind)
    else:
        tensor_ind = filename.split('_')[-1].split('.')[0]
        tensor_name = filetype + '_' + tensor_ind + '.pt'  
    
    tensor_loaded = torch.load(os.path.join(path, tensor_name))
    return tensor_loaded


def beta_retrieve(beta_value, beta_range, sample_id, num_samples, epoch_id, sparsity_flag, updated_index):
    ''' retrieve beta value
    Args:
        beta_value: current beta value before update
        beta_range: numpy array of betas
        sample_id: current sample index
        num_samples: len(train_loader)
        epoch_id: current epoch id
        sparsity_flag: torch.tensor, if all True, beta stop changing, if False, beta kept growing
        updated_index: the index for the sample, which updated beta last time
    Returns:
        beta_value: float, retrieve from beta_range.
    '''
    # not to update at the first sample first epoch
    if epoch_id*num_samples+sample_id == 0 and torch.all(sparsity_flag):
        updated_index = -1
        return beta_value, updated_index
    else: 
        # if not updated for this sample, then update_index is not updated 
        if torch.all(sparsity_flag):
            return beta_value, updated_index
        else:
            # updated for the first sample first epoch
            if epoch_id*num_samples+sample_id == 0: 
                updated_index = 0
                beta_value = beta_range[updated_index]
            else:
                updated_index += 1
                beta_value = beta_range[updated_index]
        return beta_value, updated_index 


def beta_apply(beta, sparsity_flags, beta_range, sample_id, num_samples, epoch_id, updated_index, agreement_threshold, device):
    ''' update beta value w.r.t. sample_id, epoch_id and sparsity flags

    beta increase independently for each map, if map is deemed to be updated, it updated from the last update index
    after the first epoch, stop changing if over 70% samples, supports stop

    Args:
        beta: torch.tensor to be updated, shape [num_gpus, num_stacks, num_maps]
        sparsity_flags: list of list of tensors, each flag repeat num_gpu times, refer to beta_retrieve for more info
        beta_range: np.array of linearly spaced values as growing betas
        sample_id: No. of batch, start from 0
        num_samples: len(train_loader)
        epoch_id: No. of epoch, start from 0
        device: gpu or cpu
        update_index: shape [num_stacks, num_maps]
        agreement_threshold: the percentage of agreeement expected to disable updating
    Returns:
        beta: updated beta, same shape, -1. is the stop sign
    '''
    for stack_id, stack_flag in enumerate(sparsity_flags): 
        for map_id, map_flag in enumerate(stack_flag):
            # override sparsity flag if updating only happens for few samples
            if epoch_id >= 1:
                # print(f'stack {stack_id} map {map_id}: agreemnt rate {updated_index[stack_id, map_id].item() / (epoch_id*num_samples+sample_id)}')
                if updated_index[stack_id, map_id].item() / (epoch_id*num_samples+sample_id) < agreement_threshold and updated_index[stack_id, map_id].item() / (epoch_id*num_samples+sample_id) >= 0:
                    map_flag[...] = 1. # stop changing
                    # print(f'stack {stack_id} map {map_id} stop updating')

            beta[:, stack_id, map_id], updated_index[stack_id, map_id] = beta_retrieve(beta[:, stack_id, map_id], beta_range, sample_id, num_samples, epoch_id, map_flag, updated_index[stack_id, map_id].item()) 
    return beta, updated_index
    
    # final_flag = torch.zeros_like(sparsity_flags).fill_(-1.).to(device)
  
    # if torch.equal(sparsity_flags, final_flag): # stop changing if all betas reaches stop sign
    #     return beta
    # else:
    #     for stack_id, stack_flag in enumerate(sparsity_flags): # beta can still change if one of them still did not reach stop sign
    #         for map_id, map_flag in enumerate(stack_flag):
    #             beta[:, stack_id, map_id] = beta_retrieve(beta[:, stack_id, map_id], beta_range, sample_id, num_samples, epoch_id, map_flag) 
    # return beta

# for stack_id, stack_flag in enumerate([]):
#     for map_id, map_flag in enumerate(stack_flag):
#         print(stack_id)
#         print(map_id)