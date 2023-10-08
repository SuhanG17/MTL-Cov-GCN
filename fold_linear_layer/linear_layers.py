# refer to repo: https://github.com/cure-lab/LTSF-Linear
# config.individal 是指是否给每一维度单独一个线性层。True和False都要试一下。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    return net

## linear model
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.src_seq_len
        self.pred_len = configs.trg_seq_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.channels = configs.input_dim
        self.channels = configs.sensors
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def if_by_channel(self, x):
        # x: [Batch, Input length, Sensors(channels)]
        if self.individual:
                output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        
        return x # [Batch, Input length, Sensors(channels)]


    def forward(self, x):
        # x: [Batch, Input length, Sensors(channels), num_vars] 
        num_vars = x.shape[-1]
        if num_vars == 1:
            x = x.squeeze(-1) # [Batch, Input length, Sensors(channels), 1]  -> [Batch, Input length, Sensors(channels)] 
            x = self.if_by_channel(x)
            x = x.unsqueeze(-1) # [Batch, Input length, Sensors(channels)] -> [Batch, Input length, Sensors(channels), 1] 
        else:
            B, _, C, _ = x.shape
            out = torch.zeros(B, self.pred_len, C).to(x.device) # [Batch, Output length, Sensors(channels)]
            for var in range(num_vars):
                x_slice = x[..., var].squeeze(-1) # [Batch, Input length, Sensors(channels), num_vars] -> [Batch, Input length, Sensors(channels), 1] -> [Batch, Input length, Sensors(channels)]  
                out += self.if_by_channel(x_slice)
            x = out.unsqueeze(-1)
            
        return x # [Batch, Output length, Sensors(channels), 1] 


## DLinear
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLModel, self).__init__()
        self.seq_len = configs.src_seq_len
        self.pred_len = configs.trg_seq_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.sensors

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
    
    def if_by_channel(self, x):
        # x: [Batch, Input length, Sensors(channels)]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel] # [Batch, Input length, Sensors(channels)]


    def forward(self, x):
         # x: [Batch, Input length, Sensors(channels), num_vars] 
        num_vars = x.shape[-1]
        if num_vars == 1:
            x = x.squeeze(-1) # [Batch, Input length, Sensors(channels), 1]  -> [Batch, Input length, Sensors(channels)] 
            x = self.if_by_channel(x)
            x = x.unsqueeze(-1) # [Batch, Input length, Sensors(channels)] -> [Batch, Input length, Sensors(channels), 1] 
        else:
            # # only output incidence            
            # B, _, C, _ = x.shape
            # out = torch.zeros(B, self.pred_len, C).to(x.device) # [Batch, Output length, Sensors(channels)]
            # for var in range(num_vars):
            #     x_slice = x[..., var].squeeze(-1) # [Batch, Input length, Sensors(channels), num_vars] -> [Batch, Input length, Sensors(channels), 1] -> [Batch, Input length, Sensors(channels)]  
            #     out += self.if_by_channel(x_slice)
            # x = out.unsqueeze(-1) # [Batch, Output length, Sensors(channels), 1] 

            # output all
            outs = []
            for var in range(num_vars):
                x_slice = x[..., var].squeeze(-1) # [Batch, Input length, Sensors(channels), num_vars] -> [Batch, Input length, Sensors(channels), 1] -> [Batch, Input length, Sensors(channels)]  
                out = self.if_by_channel(x_slice)
                outs.append(out)
            x = torch.stack(outs, dim=-1) # [Batch, Output length, Sensors(channels)] -> [Batch, Output length, Sensors(channels), num_vars]
            
        return x 

# class DLModel(nn.Module):
#     """
#     Decomposition-Linear
#     """
#     def __init__(self, configs):
#         super(DLModel, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len

#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.channels = configs.enc_in

#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
            
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

#                 # Use this two lines if you want to visualize the weights
#                 # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#                 # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
#             # Use this two lines if you want to visualize the weights
#             # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#             # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
#                 trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)

#         x = seasonal_output + trend_output
#         return x.permute(0,2,1) # to [Batch, Output length, Channel]

## NLinear
class NLModel(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(NLModel, self).__init__()
        self.seq_len = configs.src_seq_len
        self.pred_len = configs.trg_seq_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.sensors
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
    
    def if_by_channel(self, x):
        # x: [Batch, Input length, Sensors(channels)]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last 
        return x # [Batch, Input length, Sensors(channels)]

    def forward(self, x):
        # x: [Batch, Input length, Sensors(channels), num_vars] 
        num_vars = x.shape[-1]
        if num_vars == 1:
            x = x.squeeze(-1) # [Batch, Input length, Sensors(channels), 1]  -> [Batch, Input length, Sensors(channels)] 
            x = self.if_by_channel(x)
            x = x.unsqueeze(-1) # [Batch, Input length, Sensors(channels)] -> [Batch, Input length, Sensors(channels), 1] 
        else:
            B, _, C, _ = x.shape
            out = torch.zeros(B, self.pred_len, C).to(x.device) # [Batch, Output length, Sensors(channels)]
            for var in range(num_vars):
                x_slice = x[..., var].squeeze(-1) # [Batch, Input length, Sensors(channels), num_vars] -> [Batch, Input length, Sensors(channels), 1] -> [Batch, Input length, Sensors(channels)]  
                out += self.if_by_channel(x_slice)
            x = out.unsqueeze(-1)
            
        return x # [Batch, Output length, Sensors(channels), 1] 