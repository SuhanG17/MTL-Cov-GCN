import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
# print(torch.__version__)

# GCNConv Related
from torch_sparse import SparseTensor
from GCN_conv import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import Linear

# other packages for transformer
import numpy as np
import math, copy

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Device: {}'.format(device))

""" Graph Neural Network"""
class GConvLayer(nn.Module):
    ''' create gcn layer regardless of number of maps
    Notice that we use many loops in this class because we want to avoid mention number of convolution types explicitly.

    Attributes:
        input_dim: input dim of x
        hidden_dim: hidden dim for gcn layer
        output_dim: output dim for x, for encoder, output dim = hidden dim
        num_layers: number of layers for gcn, same as the order, or the hop size for neighborhood
        dropout: dropout rate
        bn: if apply batch_normalization in gcn_layer
        conv_type: choose between 'SAGE', 'GCN'
        num_maps: number of sensor maps, including map comes with data, if used.
        symmetrics: a list containing booleans of sensor map symmetry. 
                    The order of entry matters: [original map, adp map1, adp map2, ...] or [adp map1, adp map2, ...]
                    if SAGEConv, symmetrics became redundant param;
                    symmetrics is the same as undirected;
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bn, conv_type, num_maps, symmetric=[]):
        super(GConvLayer, self).__init__()
        self.dropout = dropout
        self.bn = bn
        self.concat = True 

        self.hidden_list = nn.ModuleList([])

        for i in range(num_maps):
           self.hidden_list.append(self.build_convs(conv_type, input_dim, hidden_dim, num_layers, symmetric[i])) 

        if self.bn:
            self.bn_list = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_dim) for _ in range(num_maps)])

        if self.concat:
            self.output_layer = Linear(hidden_dim*num_maps, output_dim)

    def build_convs(self, conv_type, input_dim, hidden_dim, num_layers, symmetric):
        ''' build convolutions for hidden layers '''
        if conv_type == 'GCN':
            convs = nn.ModuleList([GCNConv(in_channels=input_dim, out_channels=hidden_dim, symmetric=symmetric) for _ in range(num_layers)])
        elif conv_type == 'SAGE':
            convs = nn.ModuleList([SAGEConv(in_channels=input_dim, out_channels=hidden_dim) for _ in range(num_layers)])
        elif conv_type == 'GAT':
            raise NotImplementedError
        return convs

    def forward(self, x, adj_t):
        '''
        each block corresponds to a map
        inside each block, for hop=2
        x0 -> [conv0] -> x1
        x0 -> [conv1] -> x2
        if bn:
            x = ReLU(bn(x1+x2))
        else:
            x = ReLU(x1+x2)
        
        concat means the outputs from graphs are concatenated in the last dim and a linear layer is used to match the dim
        '''
        if self.bn:
            gcn_outputs = []
            for i, (block, bn_layer) in enumerate(zip(self.hidden_list, self.bn_list)):
                block_outputs = []
                for layer in block:
                    block_outputs.append(layer(x, adj_t[i])) # [batch_size*seq_len, sensors, dim]
                out = bn_layer(torch.stack(block_outputs, dim=0).sum(0).transpose(1, 2)) # [batch_size*seq_len, sensors, dim] -> [batch_size*seq_len, dim, sensors]
                out = F.relu(out).transpose(1, 2) # [batch_size*seq_len, dim, sensors] -> [batch_size*seq_len, sensors, dim] 
                gcn_outputs.append(out)
        else:
            gcn_outputs = []
            for i, block in enumerate(self.hidden_list):
                block_outputs = []
                for layer in block:
                    block_outputs.append(layer(x, adj_t[i])) # [batch_size*seq_len, sensors, dim]
                out = F.relu(torch.stack(block_outputs, dim=0).sum(0)) # [batch_size*seq_len, sensors, dim]
                gcn_outputs.append(out)
        
        if self.concat:
            x = F.relu(self.output_layer(torch.cat(gcn_outputs, dim=-1)))
        else:
            x = F.relu(torch.stack(gcn_outputs, dim=0).sum(0))

        if self.training:
            x = F.dropout(x, p=self.dropout)
        
        return x # [batch_size*seq_len, sensors, dim] 


class GConv(nn.Module):
    ''' build GCN layer with attn_s 
    Attributes:
        supports: the sensory map accompanied with data, shape [sensors, sensors]
        set_diag: set supports with self-loop, redundant by gcnconv(set_self_loops); list, len=num_maps
        undirected: if true, the adjacency matrix will be transformed into symmetric matrix; list, len=num_maps
        truncate: if true, clamp the adjacency matrix w.r.t. pre-set threshold; list, len=num_maps
        threshold: pre-set threshold; list, len=num_maps
        sparsity_ratio: percent connection kept for attention maps, float out of 1.; list, len=num_maps   
        num_layers: similar to order, number of hops to go in aggregation
        dropout: dropout rate
        bn: bool, if batch_norm data in gcn
        conv_type: which gnn to use; string
        num_maps: number of maps to aggregate, depends on number of att_s calculated in transformer
        adp_supp_len: num of learnt attnetion maps for sensors, default 1 for encoder, 2 for decoder
    '''
    def __init__(self, d_model, h, sensors, supports, set_diag, undirected, truncate, threshold, sparsity_ratio, num_layers, dropout, bn, conv_type, num_maps, adp_supp_len):
        super(GConv, self).__init__()
        self.sensors = sensors 
        self.set_diag = set_diag
        self.undirected = undirected
        self.truncate = truncate
        self.threshold = threshold
        self.sparsity_ratio = sparsity_ratio

        # for situations when we only want original supports not adapted
        self.adp_supports = False
        if adp_supp_len > 0. :
            self.adp_supports = True

        if (supports is not None):
            self.supports = supports
        else:
            self.supports = None

        self.gcn_conv = GConvLayer(d_model, d_model, d_model, num_layers, dropout, bn, conv_type, num_maps, undirected)
    
    def get_sparse_mat(self, A, set_diag=False, undirected=True, truncate=False, threshold=1e-2, sparsity_monitor=True, sparsity_ratio=None):
        ''' make SparseTensor from dense adjacency matrix 

        1. undirected and truncate can be either, both or neither used. Order of operation does not matter.
        2. why not adj = adj.to_symmetric() when adj is SparseTensor? it contains in-place operation, cannot be used in training  
        '''
        # if sparsity_ratio != 1: 
        #     print(f'sensor map before any function applied \n{A}')
        if undirected:
            A = self.make_symmetric(A)
            # if sparsity_ratio != 1: 
            #     print(f'after make symmetric \n{A}')
        if truncate:
            A = self.clamp(A, threshold)
            # if sparsity_ratio != 1: 
            #     print(f'after truncate \n{A}')
        adj = SparseTensor.from_dense(A)
        adj = adj.t() # in case of directed map
        if set_diag:
            adj = adj.set_diag()   # Add diagonal entries
        if sparsity_monitor:
            return adj, self.sparsity_monitor(sparsity_ratio, A)
        else:
            return adj
    
    def make_symmetric(self, mat):
        '''make a dense matrix symmetric 
        why torch.tril() not torch.triu()?
        make_symmetric should be applied on transposed matrix, hense the lower triangle should be used, not the upper.
        '''
        lower = torch.tril(mat) # lower triangle, with diagnol
        upper = torch.tril(mat, diagonal=-1).t() # lower triangle, without diagnol, and then tranposed to become upper triangle
        adj = lower + upper
        return adj

    def clamp(self, mat, threshold):
        '''if element < threshold, set to zero''' 
        adj = torch.abs(mat)
        adj_mask = torch.zeros_like(adj)
        adj = torch.where(adj > threshold, adj, adj_mask)
        return adj
    
    def sparsity_monitor(self, set_ratio, clamp_map):
        '''  check if sparsity meets pre-set ratio
        Args:
            set_ratio: pre-set ratio, determined by user, range [0, 1], percentage to keep
            clamp_map: sparse sensor map, shape [sensors, sensors], clamped w.r.t. pre-set threshold
        Returns:
            flag: if current sparsity meet the pre-set ratio
                  caution that if num_gpus > 1, flag returns shape [num_gpus]
                  1. indicates meet the ratio, 0. indicates otherwise
        '''
        current_keep_ratio = (clamp_map > 0.).sum() / (self.sensors*self.sensors)
        current_keep_ratio = current_keep_ratio.item()
        # print(f'current_keep_ratio {current_keep_ratio}')

        flag = torch.ones(1).fill_(float(current_keep_ratio <= set_ratio))

        return flag.to(clamp_map.device)

    def reshape_mat(self, mat):
        _, sensors, _, dim = mat.shape # [batch_size, sensors, seq_len, dim]
        return mat.transpose(0, 1).reshape(sensors, -1, dim).transpose(0, 1) # [batch_size*seq_len, sensors, dim]

    def inverse_reshape_mat(self, mat, batch_size, seq_len):
        _, sensors, dim = mat.shape # [batch_size*seq_len, sensors, dim] 
        return mat.reshape(batch_size, seq_len, sensors, dim).transpose(1, 2) #[batch_size, sensors, seq_len, dim]

    def get_sensor_map_dense_sparse(self, sensor_map, sparsity_ratio, set_diag, undirected, truncate, threshold):
        ''' produce dense and sparse sensor map
        Args:
            sensor_map: [sensors, sensors]
            sparsity_ratio: a number between [0, 1], 1. indicates no sparsity is created, original map used
        '''
        if sparsity_ratio == 1.:
            # print('processing supports from data...')
            sensor_map_sparse = self.get_sparse_mat(sensor_map, set_diag=set_diag, undirected=undirected, 
                                                    truncate=truncate, threshold=threshold, sparsity_monitor=False, sparsity_ratio=sparsity_ratio)            
            return sensor_map, sensor_map_sparse 
        else:
            # print('processing learned supports...')
            sensor_map_sparse, sparsity_flag = self.get_sparse_mat(sensor_map, set_diag=set_diag, undirected=undirected, 
                                                    truncate=truncate, threshold=threshold, sparsity_monitor=True, sparsity_ratio=sparsity_ratio)
            return sensor_map, sensor_map_sparse, sparsity_flag
    
    def forward(self, x, attn_s:list):
        """ aggregate information from neighbors 
            only at testing stage, dense supports are saved
            at training stage, sparsity flags are saved
        Args:
            x: hidden feature of input; shape [batch_size, sensors, seq_len, dim]
            attn_s: a list of attention maps for sensors; element shape attn_s[0]: [batch_size, h, sensors, sensors] 

        Returns:
            x: shape unchanged, but gathered information from neighboring nodes
            sparsity_flags: lists of sparsity flags for all supports in current stack, only for training, only for adapted maps
            dense_supports: lists of supports, only for testing
        """
        # print(x.shape)
        batch_size, _, seq_len, _ = x.shape
        
        x = self.reshape_mat(x) # [batch_size, sensors, seq_len, dim] -> [batch_size, dim, sensors, seq_len]
        supports = [] # store SparseTensor

        if self.training:
            sparsity_flags = [] # store sparsity_flags for adapted maps only
        else:
            dense_supports = [] # store dense tensor at testing stage

        if (self.supports is not None):
            sensor_map = self.supports.to(x.device)  # the to(x.device) make sure that the supports are on same device with x, using DataParallel
            sensor_map, sensor_map_sparse = self.get_sensor_map_dense_sparse(sensor_map, self.sparsity_ratio[0], self.set_diag[0], self.undirected[0], self.truncate[0], self.threshold[0])
            supports.append(sensor_map_sparse)          

            if not self.training:
                dense_supports.append(sensor_map)
                
        if self.adp_supports:
            for i, at in enumerate(attn_s):
                # print(f'attn_s has len {len(attn_s)}')
            # for i, (conv, at) in enumerate(zip(self.att_conv, attn_s)):
                # mean over batch size
                # sensor_map = F.relu(conv(at)).squeeze(1).mean(dim=0) 
                # [batch_size, h, sensors, sensors] -> [batch_size, 1, sensors, sensors] -> [batch_size, sensors, sensors] -> [sensors, sensors]

                # max over batch size
                # sensor_map = F.relu(at).squeeze(1).max(dim=0)[0]
                # mean over batch size
                # sensor_map = F.relu(at).squeeze(1).mean(dim=0) 
                # just the first 1 in batch
                sensor_map = F.relu(at).squeeze(1)[0]
                # [batch_size, 1, sensors, sensors] -> [batch_size, sensors, sensors] -> [sensors, sensors]

                # torch.save(at, '/root/autodl-tmp/graph_transformer/ml_6_newsoftmax_diagnose_3/atts.pt')
                # print(f'At att_s {i}, with sparsity_ratio {self.sparsity_ratio[i]}')
                # # if self.sparsity_ratio[i] != 1.: 
                # print(f'attention map has shape {at.shape}')
                # print(f'attension map for head 0 is : \n{at[0, 0, ...]}')
                # print(f'attension map for head 1 is : \n{at[0, 1, ...]}')
                # print(f'attension map for head 2 is : \n{at[0, 2, ...]}')
                # print(f'attension map for head 3 is : \n{at[0, 3, ...]}')

                if (self.supports is not None): # if original map provided, increment from 1 not 0
                    i += 1
                
                # sensor_map = F.softmax(sensor_map, dim=1)
                # print(f'At att_s {i}, with sparsity_ratio {self.sparsity_ratio[i]}')
                # print(f'sensor map {sensor_map}')
                if self.sparsity_ratio[i] == 1.:
                    sensor_map, sensor_map_sparse = self.get_sensor_map_dense_sparse(sensor_map, self.sparsity_ratio[i], self.set_diag[i], self.undirected[i], self.truncate[i], self.threshold[i])
                    sparsity_flag = torch.ones(1).fill_(1.0).to(sensor_map.device) 
                else:
                    sensor_map, sensor_map_sparse, sparsity_flag = self.get_sensor_map_dense_sparse(sensor_map, self.sparsity_ratio[i], self.set_diag[i], self.undirected[i], self.truncate[i], self.threshold[i])
                supports.append(sensor_map_sparse) 

                if self.training:
                    sparsity_flags.append(sparsity_flag)
                else:
                    dense_supports.append(sensor_map)  

        x = self.gcn_conv(x, supports)
        # m = self.inverse_reshape_mat(x, batch_size, seq_len)

        if self.training:
            return self.inverse_reshape_mat(x, batch_size, seq_len), sparsity_flags
        else:
            return self.inverse_reshape_mat(x, batch_size, seq_len), dense_supports

""" Transformer """
class EncoderDecoder(nn.Module):
    '''A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    '''
    def __init__(self, encoder, decoder, src_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        # self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, src_mask, beta):
        '''Take in and process masked src and target sequences.
        
        Args:
            src: shape [batch_size, sensors, src_seq_len, src_dim]
            trg: shape [batch_size, sensors, trg_seq_len, trg_dim] with start_token padded and last element removed
            src_mask: shape [batch_size, sensors, 1, seq_len]
            trg_mask: shape [batch_size, sensors, target_seq_len, target_seq_len]
            beta: shape [num_gpus, N, enc_att_num + dec_att_num]

        Returns:
           self.generator(pred): predicted sequence [batch_size, sensors, target_seq_len, trg_dim] 
           sr_enc: encoder sparsity flags, list of list of tensors
           sr_dec: decoder sparsity flags, list os list of tensors
           supp_enc: encoder supports tensors, list of list of tensors 
           supp_dec: decoder supports tensors, list of list of tensors
        '''
        if self.training:
            memory, sr_enc = self.encode(src, src_mask, beta[:, :, 0]) # memory: [batch_size, sensors, src_seq_len, d_model]
            pred = self.decode(memory) # pred: [batch_size, sensors, trg_seq_len, d_model] 
            return self.generator(pred), sr_enc
        else:
            memory, supp_enc = self.encode(src, src_mask, beta[:, :, 0]) # memory: [batch_size, sensors, src_seq_len, d_model]
            pred = self.decode(memory) # pred: [batch_size, sensors, trg_seq_len, d_model] 
            return self.generator(pred), supp_enc

    def encode(self, src, src_mask, beta):
        return self.encoder(self.src_embed(src), src_mask, beta)
    
    def decode(self, memory):
        return self.decoder(memory)

class LinearDecoder(nn.Module):
    def __init__(self, src_seq_len, trg_seq_len):
        super(LinearDecoder, self).__init__()
        self.proj = nn.Linear(src_seq_len, trg_seq_len)
    
    def forward(self, x):
        out = self.proj(x.permute(0, 1, 3, 2))
        return out.permute(0, 1, 3, 2)


class Generator(nn.Module):
    "Define standard linear step to make output the same dimension as input and target."
    def __init__(self, d_model, target_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, target_dim)

    def forward(self, x):
        return self.proj(x)

""" Helper fn """
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(trg, pad, device):
    "Create a mask to hide padding and future words."
    trg_mask = (trg != pad).unsqueeze(-2)
    trg_mask = trg_mask & subsequent_mask(trg.size(-1)).to(device)
    return trg_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""# send to multiple GPUs"""
def train_vs_test(net, load_net_path, device, train_mode):
    # train from scratch
    if load_net_path == None:
        # use CPU
        if device == 'cpu':
            net = net.to(device) 
        else:
            # multiple-GPUs
            if torch.cuda.device_count()>1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
                net = net.to(device)
            # use one GPU
            else:
                print("Let's use", torch.cuda.device_count(), "GPU!") 
                net = net.to(device)

        net = net.train()

    # load model weights
    else:
        # testing
        net = net.to(device)
        net.load_state_dict(torch.load(load_net_path, map_location=torch.device(device)))

        # resume training + multi-gpu testing
        if device == 'cpu':
            net = net.to(device)
        else:
            if torch.cuda.device_count()>1:
                if train_mode:
                    print("Let's use", torch.cuda.device_count(), "GPUs to resume training!")
                else:
                    print("Let's use", torch.cuda.device_count(), "GPUs to test!")
                net = nn.DataParallel(net)
                net = net.to(device)
            # use one GPU 
            else:
                if train_mode:
                    print("Let's use one gpu to resume training!")
                else:
                    print("Let's use one gpu to test!")
                net = net.to(device) 

        if train_mode:       
            net = net.train()
        else:
            net = net.eval()

        # # resume training
        # if train_mode:
        #     if device == 'cpu':
        #         net = net.to(device)
        #         net = net.train()
        #     else:
        #         if torch.cuda.device_count()>1:
        #             print("Let's use", torch.cuda.device_count(), "GPUs to resume training!")
        #             net = nn.DataParallel(net)
        #             net = net.to(device)
        #         # use one GPU 
        #         else:
        #             print("Let's use one gpu to resume training!")
        #             net = net.to(device) 
        #     net = net.train()
        # else:
        #     net = net.eval()

    return net


""" Encoder """
class EncoderGCN(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, gcn_layer=None):
        super(EncoderGCN, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.N = N
        if gcn_layer:
            self.gcn_layers = clones(gcn_layer, N)
        else:
            self.gcn_layers = None
        
    def forward(self, x, mask, beta):
        ''''Pass the input (and mask) through each layer in turn.
        During training, model tune beta, thus return sparsity_flags;
        During testing, visualize supports used, thus, return supp_list.
        '''
        beta = beta.reshape(self.N, -1) # [1, N] -> [N, 1]        
        # print(f'encoder gcn beta {beta}, beta shape {beta.shape}')
        if self.training:
            sparsity_flags = []
            if self.gcn_layers:
                for layer, gcn_layer, b in zip(self.layers, self.gcn_layers, beta):
                    # print(f'encoder gcn b {b}, shape {b.shape}')
                    x_temporal = layer(x, mask, b)
                    x_spatial, sparsity_flag = gcn_layer(x_temporal, [layer.self_attn.attn_s])
                    x = x_temporal + x_spatial
                    sparsity_flags.append(sparsity_flag)
            else:
                for layer, b in zip(self.layers, beta):
                    x = layer(x, mask, b)
            return self.norm(x), sparsity_flags
        else: # only at testing stage, we need to save supports for visualization
            supp_list = []
            if self.gcn_layers:
                for layer, gcn_layer, b in zip(self.layers, self.gcn_layers, beta):
                    # print(f'encoder gcn b {b}, shape {b.shape}')
                    x_temporal = layer(x, mask, b)
                    x_spatial, supp = gcn_layer(x_temporal, [layer.self_attn.attn_s])
                    x = x_temporal + x_spatial
                    supp_list.append(supp)
            else:
                for layer, b in zip(self.layers, beta):
                    x = layer(x, mask, b)
            return self.norm(x), supp_list


class LayerNorm(nn.Module):
    '''Construct a layernorm module (See citation for details).'''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    '''Encoder is made up of self-attn and feed forward (defined below)
    update self.beta with provided beta, if stop sign -1. is met, used currently stored self.beta
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.beta = nn.Parameter(torch.ones(1), requires_grad=False)

    def update_beta(self, beta):
        # if beta[0] != -1.:
        #    self.beta = self.beta.fill_(beta)
        self.beta[0] = beta[0]

    def forward(self, x, mask, beta):
        "Follow Figure 1 (left) for connections."
        self.update_beta(beta)
        # print(f'encoder beta {beta}, self.beta {self.beta}')
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, self.beta))
        return self.sublayer[1](x, self.feed_forward)


""" Decoder """
class DecoderGCN(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, gcn_layer=None):
        super(DecoderGCN, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if gcn_layer:
            self.gcn_layers = clones(gcn_layer, N)
        else:
            self.gcn_layers = None
        
    def forward(self, x, memory, src_mask, trg_mask, beta):
        ''' Decoder with tunable beta
        Same structure with encoder, but decoder has 2 supports for each stack
        '''
        beta = beta.squeeze(0) # beta: [1, stack, attn] -> [stack, attn]
        # print(f'decoder gcn beta {beta}, beta shape {beta.shape}')
        if self.training:
            sparsity_flags = []
            if self.gcn_layers:
                for layer, gcn_layer, b in zip(self.layers, self.gcn_layers, beta): 
                    # print(f'decoder gcn b {b}, shape {b.shape}')
                    x_temporal = layer(x, memory, src_mask, trg_mask, b)
                    x_spatial, sparsity_flag = gcn_layer(x_temporal, [layer.self_attn.attn_s, layer.src_attn.attn_s])
                    x = x_temporal + x_spatial
                    sparsity_flags.append(sparsity_flag)
            else:
                for layer, b in zip(self.layers, beta):
                    x = layer(x, memory, src_mask, trg_mask, b)
            return self.norm(x), sparsity_flags
        else:
            supp_list = []
            if self.gcn_layers:
                for layer, gcn_layer, b in zip(self.layers, self.gcn_layers, beta): 
                    # print(f'decoder gcn b {b}, shape {b.shape}')
                    x_temporal = layer(x, memory, src_mask, trg_mask, b)
                    x_spatial, supp = gcn_layer(x_temporal, [layer.self_attn.attn_s, layer.src_attn.attn_s])
                    x = x_temporal + x_spatial
                    supp_list.append(supp)
            else:
                for layer, b in zip(self.layers, beta):
                    x = layer(x, memory, src_mask, trg_mask, b)
            return self.norm(x), supp_list

class DecoderLayer(nn.Module):
    '''Decoder is made of self-attn, src-attn, and feed forward (defined below)'''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.beta = nn.Parameter(torch.ones(2), requires_grad=False)

    def update_beta(self, beta):
        for i, b in enumerate(beta):
            # if b != -1.:
            self.beta[i] = b
 
    def forward(self, x, memory, src_mask, trg_mask, beta):
        '''Follow Figure 1 (right) for connections.'''
        self.update_beta(beta)
        # print(f'decoder beta {beta}, beta shape {beta.shape}, self.beta {self.beta}') 
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask, self.beta[0]))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, self.beta[1]))
        return self.sublayer[2](x, self.feed_forward)

""" Attention """
# original attention
def attention(query, key, value, mask=None, dropout=None):
    '''Compute "Scaled Dot Product Attention"'''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# tunable softmax attention
def softmax_beta(score_matrix, beta):
    ''' control softmax values with beta
    softmax_score = exp(beta * x) / sum(exp(beta * x))
    assuming x to be positive:

    beta = 0., it's a uniform distribution, p=1/score_matrix.shape[-1]
    beta = 1., it's the default the softmax fn
    beta -> inf, it puts all probability mass on the largest prob

    The higher the beta, the more confident you are with your prediction.
    '''
    return torch.exp(beta*score_matrix) / torch.sum(torch.exp(beta*score_matrix), dim=-1, keepdim=True)

def stable_softmax_beta(score_matrix, beta):
    """ control softmax values with beta
    To ensure that softmax won't result in nan, we choose to subtract the maximum to ensure torch.exp won't overflow:
    because exp(x - c)/ Σ exp(x - c) = exp(x) / Σ exp(x)

    softmax_score = exp(beta * x) / Σ (exp(beta * x))
    assuming x to be positive:

    beta = 0., it's a uniform distribution, p=1/score_matrix.shape[-1]
    beta = 1., it's the default the softmax fn
    beta -> inf, it puts all probability mass on the largest prob

    The higher the beta, the more confident you are with your prediction.
    """
    # print(f'softmax score_matrix {score_matrix.device}, beta {beta.device}')
    col_max = torch.max(score_matrix, axis=-1)[0].unsqueeze(-1) # shape [sensors, 1]
    numerator = torch.exp((score_matrix - col_max)*beta)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator

def attention_beta(query, key, value, beta, mask=None, dropout=None):
    '''Compute "Scaled Dot Product Attention" with softmax, whose softness is tunable'''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn = F.softmax(scores, dim = -1)
    # p_attn = softmax_beta(scores, beta)
    p_attn = stable_softmax_beta(scores, beta)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttentionHighDim(nn.Module):
  def __init__(self, h, d_model, dropout=0.1, sensor_map=True):
    super(MultiHeadedAttentionHighDim, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.linears_s = clones(nn.Linear(d_model, d_model), 3)
    self.attn = None
    self.attn_s = None
    self.dropout = nn.Dropout(p=dropout)
    self.sensor_map = sensor_map

  def forward(self, query, key, value, mask=None, beta=None):
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(2)
    nbatches, sensors = query.size(0), query.size(1)
    # print(f'query has shape {query.shape}, key has shape {key.shape}, value has shape {value.shape}')
    # query, key, value: [batch_size, sensors, seq_len, d_model]

    if self.sensor_map:
        # 4) Apply attention to sensor dim
        sensor_h = 1
        sensor_d_k = self.d_k * self.h
        
        query_s, key_s, value_s = [l(x).view(nbatches, sensors, -1, sensor_h, sensor_d_k).permute(0, 3, 1, 2, 4).reshape(nbatches, sensor_h, sensors, -1)
                                    for l, x in zip(self.linears_s, (query, key, value))]

        # [batch_size, sensors, seq_len, d_model]
        # [batch_size, sensors, seq_len, sensor_h, sensor_d_k] -> [batch_size, sensor_h, sensors, seq_len, sensor_d_k] -> [batch_size, sensor_h, sensors, seq_len * d_k]

        # decoding stage, key and value has different seq_len, compared to query, hence, the slicing below
    #   _, self.attn_s = attention(query_s, key_s[..., :query_s.shape[-1]], value_s[..., :query_s.shape[-1]], mask=None, dropout=self.dropout)
        _, self.attn_s = attention_beta(query_s, key_s[..., :query_s.shape[-1]], value_s[..., :query_s.shape[-1]], beta=beta, mask=None, dropout=self.dropout)


    # 1) Do all the linear projections in batch from d_model => h x d_k 
    query, key, value = \
            [l(x).view(nbatches, sensors, -1, self.h, self.d_k).transpose(2, 3) #[batch_size, sensors, h, seq_len, d_k]
             for l, x in zip(self.linears, (query, key, value))]

    # 2) Apply attention on all the projected vectors in batch. 
    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
    # x: #[batch_size, sensors, h, seq_len, d_k]
    # attn: #[batch_size, sensors, h, seq_len, seq_len]

    # 3) "Concat" using a view and apply a final linear. 
    x = x.transpose(2, 3).contiguous() \
          .view(nbatches, sensors, -1, self.h * self.d_k)

    # if self.sensor_map:
    #   # 4) Apply attention to sensor dim
    #   # query_s, key_s, value_s = query.transpose(1, 2).reshape(nbatches, self.h, sensors, -1), key.transpose(1, 2).reshape(nbatches, self.h, sensors, -1), value.transpose(1, 2).reshape(nbatches, self.h, sensors, -1)
    #   query_s, key_s, value_s = [l(x).view(nbatches, sensors, -1, self.h, self.d_k).permute(0, 3, 1, 2, 4).reshape(nbatches, self.h, sensors, -1)
    #                              for l, x in zip(self.linears_s, (query.transpose(2, 3).reshape(nbatches, sensors, -1, self.h*self.d_k), key.transpose(2, 3).reshape(nbatches, sensors, -1, self.h*self.d_k), value.transpose(2, 3).reshape(nbatches, sensors, -1, self.h*self.d_k)))]

    #   # [batch_size, sensors, h, seq_len, d_k] -> [batch_size, sensors, seq_len, h, d_k] -> [batch_size, sensors, seq_len, h * d_k]
    #   # [batch_size, sensors, seq_len, h, d_k] -> [batch_size, h, sensors, seq_len, d_k] -> [batch_size, h, sensors, seq_len * d_k]

    #   # decoding stage, key and value has different seq_len, compared to query, hence, the slicing below
    # #   _, self.attn_s = attention(query_s, key_s[..., :query_s.shape[-1]], value_s[..., :query_s.shape[-1]], mask=None, dropout=self.dropout)
    #   _, self.attn_s = attention_beta(query_s, key_s[..., :query_s.shape[-1]], value_s[..., :query_s.shape[-1]], beta=beta, mask=None, dropout=self.dropout)

    return self.linears[-1](x) # [batch_size, sensors, seq_len, d_model]

""" Position-Wise FeedForward """
class PositionwiseFeedForward(nn.Module):
    '''Implements FFN equation.'''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

""" Linear Layer to set dim """
class DimMatch(nn.Module):
    def __init__(self, d_model, dim_of_input):
      ''' Implement dimension change for coninuous data (dim_of_input can handel both input_dim and target_dim) '''
      super(DimMatch, self).__init__()
      self.lut = nn.Linear(dim_of_input, d_model)
      self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

""" Positional Encoding """
class PositionalEncoding(nn.Module):
    '''Implement the PE function.'''
    def __init__(self, d_model, sensors, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.sensors = sensors

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # because we have sensor dimention, positional encoding for a sensor need to be repeated K times, where K = num_sensors
        x = x + self.pe[:, :x.size(2)].unsqueeze(1).repeat(1, self.sensors, 1, 1)
        return self.dropout(x)


""" Full Model """
def init_weights(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.kaiming_normal_(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, GCNConv):
        m.reset_parameters()
    if isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()
    if isinstance(m, nn.Conv2d):
        # torch.nn.init.kaiming_normal_(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def build_network(src_seq_len, trg_seq_len, input_dim, target_dim, N=6,  
                  d_model=512, d_ff=2048, h=8, dropout=0.1, 
                  add_gcn=True, sensors=10, supports=None, 
                  set_diag=[[True], [True, True]], undirected=[[True], [True, True]], truncate=[[True], [True, True]], threshold=[[1e-2], [1e-2, 1e-2]], sparsity_ratio=[[0.4], [0.4, 0.4]],
                  num_layers=2, bn=False, conv_type=['GCN', 'GCN'], num_maps=[1, 2], adp_supp_len=[1, 2],
                  device='cpu', train_mode=True, load_net_path=None):
    '''Helper: Construct a model from hyperparameters.'''
    c = copy.deepcopy
    attn = MultiHeadedAttentionHighDim(h, d_model, sensor_map=add_gcn)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, sensors, dropout)
    if add_gcn:
        gcn_enc = GConv(d_model, h, sensors, supports, set_diag[0], undirected[0], truncate[0], threshold[0], sparsity_ratio[0], num_layers, dropout, bn, conv_type[0], num_maps[0], adp_supp_len[0])
        # gcn_dec = GConv(d_model, h, sensors, supports, set_diag[1], undirected[1], truncate[1], threshold[1], sparsity_ratio[1], num_layers, dropout, bn, conv_type[1], num_maps[1], adp_supp_len[1])
    else:
        gcn_enc = None
        gcn_dec = None
    
    gcn_dec = None
    model = EncoderDecoder(
        EncoderGCN(EncoderLayer(d_model, c(attn), c(ff), dropout), N, c(gcn_enc)),
        # DecoderGCN(DecoderLayer(d_model, c(attn), c(attn), 
        #                      c(ff), dropout), N, c(gcn_dec)),
        LinearDecoder(src_seq_len, trg_seq_len),
        nn.Sequential(DimMatch(d_model, input_dim), c(position)),
        # nn.Sequential(DimMatch(d_model, target_dim), c(position)),
        Generator(d_model, 1)) #target_dim))
    
    model.apply(init_weights)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    return train_vs_test(model, load_net_path, device, train_mode)


""" Greedy Decode """
# def batch_greedy_decode(model, src, src_mask, max_len, ys, beta, device='cpu'):
#     """ Batched version decode output one by one
#     max_len: seq length of target, without start token
#     ys: the start of ys containing start_token only, shape [batch_size, sensors, 1, 1]
#     beta: use self.data in model, hence the -1. flag
#           [[encoder_layer_1, decoder_layer_1_self_attn, decoder_layer_1_src_attn], [encoder_layer_2, decoder_layer_2_self_attn, decoder_layer_2_src_attn]]
#     """
#     memory, supp_enc = model.encode(src, src_mask, beta[:, :, 0])
#     #   batch_size = src.size(0)
#     #   ys = torch.ones(batch_size, sensors, 1, 1).fill_(start_symbol).type_as(src.data)
#     supp_dec_list = []
#     for _ in range(max_len):
#         out, supp_dec = model.decode(memory, src_mask, 
#                                      ys, 
#                                      make_std_mask(ys.squeeze(-1), 0, device),
#                                      beta[:, :, 1:])
#         supp_dec_list.append(supp_dec)
#         pred_one_step = model.generator(out[:, :, -1, :])
#         ys = torch.cat([ys, pred_one_step.unsqueeze(-1)], dim=2)
#     return ys[:, :, 1:, :], supp_enc, supp_dec_list

def module_to_parallel(module, device, train_mode):
    if torch.cuda.device_count()>1:
        module = nn.DataParallel(module)
        module.to(device)
    else:
        module.to(device)

    if train_mode:
        return module.train()
    else:
        return module.eval()
        

def batch_greedy_decode(model, src, src_mask, max_len, ys, beta, device='cpu'):
    """ Batched version decode output one by one, enabled nn.Parallel
    max_len: seq length of target, without start token
    ys: the start of ys containing start_token only, shape [batch_size, sensors, 1, 1]
    beta: use self.data in model, hence the -1. flag
          [[encoder_layer_1, decoder_layer_1_self_attn, decoder_layer_1_src_attn], [encoder_layer_2, decoder_layer_2_self_attn, decoder_layer_2_src_attn]]
    """
    # check if model is already parallel, need it on one-gpu
    if isinstance(model, nn.DataParallel):
        model = model.module
        model = model.to(device)
    else:
        model = model.to(device)
    model = model.eval()

    # put each module to DataParallel
    # encoder
    para_src_emb = module_to_parallel(model.src_embed, device, False)
    para_enc = module_to_parallel(model.encoder, device, False)
    # decoder
    para_trg_emb = module_to_parallel(model.trg_embed, device, False)
    para_dec = module_to_parallel(model.decoder, device, False)
    # generator
    para_gen = module_to_parallel(model.generator, device, False)

    # decode by module
    memory, supp_enc = para_enc(para_src_emb(src), src_mask, beta[:, :, 0])
    supp_dec_list = []
    for _ in range(max_len):
        out, supp_dec = para_dec(para_trg_emb(ys), memory, 
                                 src_mask, 
                                 make_std_mask(ys.squeeze(-1), 0, device),
                                 beta[:, :, 1:])
        supp_dec_list.append(supp_dec)
        pred_one_step = para_gen(out[:, :, -1, :])
        ys = torch.cat([ys, pred_one_step.unsqueeze(-1)], dim=2)
    return ys[:, :, 1:, :], supp_enc, supp_dec_list 

""" Prove that there is no information leak in decoder for sensor_attention

For original attention:
query, key, value has shape [batch_size, sensors, h, seq_len, d_k]

For sensor attention:
query, key, value has shape [batch_size, h, sensors, seq_len * d_k]

The attention calculation only happens in the last two dimension, hence batch_size, h, can be ignored

To be clarify our point, we 1) overlooked division by sqrt(d_k),
                            2) did not use softmax
                            3) use integers as psudo-data.

sensors = 4
seq_len = 3
d_k = 2

Conclusion: 
during sensor attention calculation, because of reshaping, we never actually calculate dot-product between seq_len:0 to (seq_len:1, seq_len:2),
the only dot-product is (seq_len:0 @ seq_len:0), (seq_len:1 @ seq_len:1), (seq_len:2 @ seq_len:2) 
between or across sensors. Hence, information was not leaked.
"""
### uncomment from here
# sensor_1 = torch.tensor([[1, 2], [2, 3], [3, 4]])
# sensor_2 = torch.tensor([[2, 3], [3, 4], [4, 5]])
# sensor_3 = torch.tensor([[3, 4], [4, 5], [5, 6]])
# sensor_4 = torch.tensor([[4, 5], [5, 6], [6, 7]])
# sensors_all = torch.stack((sensor_1, sensor_2, sensor_3, sensor_4))

# def raw_attn(sensor):
#     seq_len = sensor.shape[1]
#     mask = subsequent_mask(seq_len) # emulate trg_mask
#     scores = torch.matmul(sensor, sensor.transpose(-2, -1))
#     scores = scores.masked_fill(mask == 0, -1e9)
#     return scores

# def raw_map_attn(sensor):
#     num_sensors, seq_len, d_k = sensor.shape
#     map_score = torch.matmul(sensor.reshape(num_sensors, seq_len*d_k), sensor.reshape(num_sensors, seq_len*d_k).transpose(-2, -1)) 
#     return map_score

# scores = temp_attn(sensors_all)
# scores
# map_score = temp_map_attn(sensors_all)
# map_score
### uncomment end here
