import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

"""# Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure (cite). Here, the encoder maps an input sequence of symbol representations $(x_1, x_2, ..., x_n)$ to a sequence of continuous representations $(z_1, z_2, ..., z_n)$. Given $\boldsymbol{z}$ , the decoder then generates an output sequence $(y_1, y_2, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive (cite), consuming the previously generated symbols as additional input when generating the next.

**Notice** that in time-series prediction, the input sequence is already continuous and hence the embedding of sequence needs to be adapted to continuous input. Embedding layer will now serve as linear layers which extract features from inputs sequence by increase the dimention of such input.

In the original repo, they have machine translation as the task, hence, vocab is the final dimension of output. In our case, the final dimension will be 1.
"""

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask) # [batch_size, seq_len, dim=d_model]
        pred = self.decode(memory, src_mask, tgt, tgt_mask) # [batch_size, seq_len, dim=d_model]
        return self.generator(pred) # [batch_size, seq_len, dim=target_dim]
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear step to make output the same dimension as input and target."
    def __init__(self, d_model, target_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, target_dim)

    def forward(self, x):
        return self.proj(x)

"""# Encoder and Decoder Stacks

## Encoder

The encoder is composed of a stack of $N=6$ identical layers.
"""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
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
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

"""## Decoder

The decoder is also composed of a stack of $N=6$ identical layers.

### Original: product one-by-one
"""

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

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

"""# Attention

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix . The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_K}})V$

"""

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # modification made to avoid tracer warning -> change python int d_k to a tensor remove the warning
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.ones(1).fill_(d_k)).to(query.device)
            #  / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

"""# Position-wise Feed-Forward Networks"""

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

"""## Linear Layer to set dim"""

class DimMatch(nn.Module):
    def __init__(self, d_model, dim_of_input):
      """Implement dimension change for coninuous data (dim_of_input can handel both input_dim and target_dim)"""
      super(DimMatch, self).__init__()
      self.lut = nn.Linear(dim_of_input, d_model)
      self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

"""# Positional Encoding

"""

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

"""# helper fns"""
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
                net = net.to(device)

        net = net.train()

    # load model weights
    else:
        # testing
        net = net.to(device)
        net.load_state_dict(torch.load(load_net_path, map_location=torch.device(device)))

        # resume training
        if train_mode:
            if device == 'cpu':
                net = net.to(device)
                net = net.train()
            else:
                if torch.cuda.device_count()>1:
                    print("Let's use", torch.cuda.device_count(), "GPUs to resume training!")
                    net = nn.DataParallel(net)
                    net = net.to(device)
                # use one GPU 
                else:
                    print("Let's use one gpu to resume training!")
                    net = net.to(device) 
            net = net.train()
        else:
            net = net.eval()

    return net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_network(input_dim, target_dim, N=6, 
                  d_model=512, d_ff=2048, h=8, dropout=0.1, 
                  device='cpu', train_mode=True, load_net_path=None):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(DimMatch(d_model, input_dim), c(position)),
        nn.Sequential(DimMatch(d_model, target_dim), c(position)),
        Generator(d_model, target_dim))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return train_vs_test(model, load_net_path, device, train_mode)

# import seaborn

"""# Test Decode

Use with original decoder
linear decoder should be used in testing as it is in training

## single decoder
"""

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """ decode output one by one
    max_len: seq length of target, without start token"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len):
        out = model.decode(memory, src_mask, 
                            ys, 
                            subsequent_mask(ys.size(1)))
        pred_one_step = model.generator(out[:, -1, :])
        ys = torch.cat([ys, pred_one_step.unsqueeze(-1)], dim=1)
    return ys[:, 1:, :]

# # example
# model.eval()
# src = torch.rand(1, 70, 1)
# src_mask = torch.ones(1, 1, 70)
# pred = greedy_decode(model, src, src_mask, max_len=56, start_symbol=1)
# print(pred.shape)

"""## batch decoder"""

def batch_greedy_decode(model, src, src_mask, max_len, start_symbol, device='cpu'):
    """ Batched version decode output one by one
    max_len: seq length of target, without start token"""
    memory = model.encode(src, src_mask)
    ys = start_symbol # [batch_size, 1, trg_dim]
    for i in range(max_len):
        out = model.decode(memory, src_mask, 
                            ys, 
                            make_std_mask(ys.mean(-1), 0, device)) # out: [batch_size, i+1, d_model]
        pred_one_step = model.generator(out[:, -1, :]) # [batch_size, 1, trg_dim]
        ys = torch.cat([ys, pred_one_step.unsqueeze(1)], dim=1)
        # print(f'out {out.shape} \npred_one_step {pred_one_step.shape} \nys {ys.shape}')
    return ys[:, 1:, :]

# # example
# model.eval()
# src = torch.rand(4, 70, 1).to(device)
# src_mask = torch.ones(4, 1, 70).to(device)
# pred = batch_greedy_decode(model, src, src_mask, max_len=56, start_symbol=1, device=device)
# print(pred.shape)

"""# visualize attention maps"""

# def draw(data, x, y, ax):
#     seaborn.heatmap(data, 
#                     xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
#                     cbar=False, ax=ax)

# src_seq_len = 70
# sent = [str(x) for x in range(src_seq_len)]
# sent = ' '.join(sent).split()

# fig, axs = plt.subplots(1,2, figsize=(20, 10))
# draw(model.encoder.layers[1].self_attn.attn[0,0].data.cpu(), sent, sent, ax=axs[0])
# draw(model.encoder.layers[1].self_attn.attn[0,1].data.cpu(), sent, [], ax=axs[1])
