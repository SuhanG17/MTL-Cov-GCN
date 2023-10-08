import torch
import torch.nn as nn
from metrics import mae
import matplotlib.pyplot as plt
import numpy as np

"""# optimizer with warmup"""
class NoamOpt:
    """ Optim wrapper that implements rate. modified to enable training resuming
    model_size: d_model param
    factor: hyper-param to control how high is the peak value
    warmup: hypter-param to control on which epoch the peak value appears
    optimizer: which optimizer to use, Adam with lr=0 as default
    resume_from_step: resume training from step k, default 0 meaning train from scratch

    """
    def __init__(self, model_size, factor, warmup, optimizer, resume_from_step=0):
        self.optimizer = optimizer
        self._step = resume_from_step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        return rate
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
        # when epoch smaller than warmup, second value selected
        # when epoch bigger than warmup, first value selected


# # under total of 500 epoch * 375 samples = 187500 steps
# d_model = 128
# num_steps = 187500
# factor = 2.4
# peak = [50000, 100000, 150000]
# opts = [NoamOpt(d_model, factor, peak[0], None, 0), 
#         NoamOpt(d_model, factor, peak[1], None, 0),
#         NoamOpt(d_model, factor, peak[2], None, 0)]
# plt.plot(np.arange(1, num_steps), [[opt.rate(i) for opt in opts] for i in range(1, num_steps)])
# plt.legend([f"{d_model}:{peak[0]}", f"{d_model}:{peak[1]}", f"{d_model}:{peak[2]}"])

'''
lr ~ 0.002
factor = 10, peak = 150000, lr = ~0.0021
factor = 8, peak = 100000, lr = ~0.0021
factor = 5, peak = 50000,  lr = ~0.002

lr ~ 0.005
factor = 25, peak = 150000, lr = ~0.005
factor = 18, peak = 100000, lr = ~0.005
factor = 13, peak = 50000,  lr = ~0.005

lr ~ 0.0005
factor = 2.4, peak = 150000, lr = ~0.00055
factor = 2, peak = 100000, lr = ~0.00055
factor = 1.3, peak = 50000,  lr = ~0.00055
'''

# under total of 500 epoch * 285 samples = 142500 steps
# d_model = 128
# num_steps = int(285*300)
# print(num_steps)
# factor = 3
# peak = [20000, 30000, 50000]
# opts = [NoamOpt(d_model, factor, peak[0], None, 0), 
#         NoamOpt(d_model, factor, peak[1], None, 0),
#         NoamOpt(d_model, factor, peak[2], None, 0)]
# plt.plot(np.arange(1, num_steps), [[opt.rate(i) for opt in opts] for i in range(1, num_steps)])
# plt.legend([f"{d_model}:{peak[0]}", f"{d_model}:{peak[1]}", f"{d_model}:{peak[2]}"])

'''
lr ~ 0.001
factor = 2, peak = 20000, lr = ~0.0012
factor = 3, peak = 30000, lr = ~0.0015
factor = 4, peak = 50000, lr = ~0.0015
'''

# # resume training 2.0
# # intended
# # before train
# train_epochs = 50 #100
# ft_epochs = 50 #100
# num_batches = 375 #375 # 188
# model_size = 128
# # before ft
# best_train_epoch = 50
# factor = 3 #2.5
# warmup = 18750

def before_train(train_epochs, ft_epochs, num_batches):
    ''' decide total epochs before train
    train_epochs: train epochs intended, precise, should be bigger than warmup round
    ft_epochs: least finetune epochs intended, change based on best train epochs
    num_batches: number of batches in each epoch
    '''
    total_epochs = train_epochs + ft_epochs
    print(f'warmup and factor should be decided using {total_epochs} epochs, {total_epochs*num_batches} steps')
    print(f'warmup should be smaller than {train_epochs*num_batches} steps')
    return total_epochs

def before_finetune(total_epochs, best_train_epoch, num_batches, factor, warmup, model_size):
    ''' decide best step and finetune epochs
    total_epochs: total number of epochs intended
    best_train_epoch: epoch of train to resume from
    num_batches: number of batches in each epoch 
    '''
    step = int(best_train_epoch * num_batches)
    lr_at_pt = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))) 
    print(f'best step {step}, lr {lr_at_pt}')
    ft_epochs = total_epochs - best_train_epoch
    return step, ft_epochs

# total_epochs = before_train(train_epochs, ft_epochs, num_batches)

# # plot
def plot_lr(model_size, peak, factor, num_batches, total_epochs):
    num_steps = int(num_batches * total_epochs)
    opts = [NoamOpt(model_size, factor, peak[0], None, 0), 
            NoamOpt(model_size, factor, peak[1], None, 0),
            NoamOpt(model_size, factor, peak[2], None, 0)]
    plt.plot(np.arange(1, num_steps), [[opt.rate(i) for opt in opts] for i in range(1, num_steps)])
    plt.legend([f"{model_size}:{peak[0]}", f"{model_size}:{peak[1]}", f"{model_size}:{peak[2]}"])


# factor = 2.5 * 20 / 26
# # factor = 2.5
# peak = [15000, 20000, 30000] 
# plot_lr(model_size, peak, factor, num_batches, total_epochs)


''' batch_size = 128, num_batches = 188 
lr ~ 0.0015
factor = 1, peak = 5000, lr = ~0.0012
factor = 1.8, peak = 10000, lr = ~0.0015
factor = 2.5, peak = 20000, lr = ~0.0015
'''

''' batch_size = 64, num_batches = 375 
lr ~ 0.0015
factor = 1.7, peak = 10000, lr = ~0.0015
factor = 2.5, peak = 20000, lr = ~0.0015
factor = 3, peak = 30000, lr = ~0.0015
'''

''' batch_size = 64, num_batches = 375, need fewer round to be trained for less overfitting to happen
lr ~ 0.0015
factor = 2.5, peak = 20000, lr = ~0.0015 (same as above)
train_epochs = 60 
ft_epochs ~ 40
'''


''' batch_size = 32, num_batches = 750, need fewer round to be trained for less overfitting to happen
lr ~ 0.0012
factor = 1.4, peak = 10000, lr = lr = ~0.0012 (same as above)
factor = 2, peak = 20000, lr = ~0.0012 (same as above)
factor = 2.5, peak = 30000, lr = ~0.0012 (same as above)

factor = 1.92, peak = 15000, lr = ~0.0014
factor = 0.96, peak = 15000, lr = ~0.0007

train_epochs = 50 
ft_epochs ~ 50
'''

# num_batches = 375
# best_train_epoch = 14
# factor = 2.6
# warmup = 20*750
# resume_step, ft_epochs = before_finetune(total_epochs, best_train_epoch, num_batches, factor, warmup, model_size)
# print(f'In fact, each run contains {train_epochs} + {ft_epochs} = {train_epochs + ft_epochs} epochs')

# best step 10500, lr 0.0008203125000000001 =》 best step 5250, lr 0.00041015625000000004


# # resume training
# factor = 3
# warmup = 30000
# model_size = 128
# step = int(69*285)

# lr_at_pt = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))) 
# print(lr_at_pt)
# # when epoch smaller than warmup, second value selected
# # when epoch bigger than warmup, first value selected

# d_model = 128
# num_steps = int(100 * 285)
# print(num_steps)
# factor = 1
# peak = [5000, 8000, 10000]
# opts = [NoamOpt(d_model, factor, peak[0], None, 0), 
#         NoamOpt(d_model, factor, peak[1], None, 0),
#         NoamOpt(d_model, factor, peak[2], None, 0)]
# plt.plot(np.arange(1, num_steps), [[opt.rate(i) for opt in opts] for i in range(1, num_steps)])
# plt.legend([f"{d_model}:{peak[0]}", f"{d_model}:{peak[1]}", f"{d_model}:{peak[2]}"])
'''
lr ~ 0.0012
factor = 1, peak = 5000, lr = ~0.0012
'''




# class NoamOpt:
#     "Optim wrapper that implements rate."
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0
        
#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
        
#     def rate(self, step = None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         return self.factor * \
#             (self.model_size ** (-0.5) *
#             min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# import matplotlib.pyplot as plt
# import numpy as np
# # Three settings of the lrate hyperparameters.
# opts = [NoamOpt(512, 1, 4000, None), 
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])

# # under total of 500 epoch
# opts = [NoamOpt(256, 0.3, 50, None), 
#         NoamOpt(256, 0.3, 100, None),
#         NoamOpt(256, 0.3, 200, None)]
# plt.plot(np.arange(1, 500), [[opt.rate(i) for opt in opts] for i in range(1, 500)])
# plt.legend(["256:50", "256:100", "256:200"])

# opt = NoamOpt(256, 0.3, 50, None)
# step = 45
# # opt.rate(123)
# for step in range(45, 52):
#     print(f'step {step}: first {(step ** (-0.5)):.3f} second {(step * 50 ** (-1.5)):.3f}')
# ans = min(step ** (-0.5), step * 50 ** (-1.5))
# 0.3 * (256 ** (-0.5) * ans)


def build_optimizer(network):
    if torch.cuda.device_count() > 1:
        return NoamOpt(network.module.src_embed[0].d_model, factor=1, warmup=400, optimizer=torch.optim.Adam(network.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 
    else:
        return NoamOpt(network.src_embed[0].d_model, factor=1, warmup=400, optimizer=torch.optim.Adam(network.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def build_optimizer_linear(network, lr, optimizer_method, factor, warmup, resume_from_step):
    if optimizer_method == 'SGD':
        return torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9)
    if optimizer_method == 'Adam':
        return torch.optim.Adam(network.parameters(), lr=lr)
    if optimizer_method =='AdamX':
        return torch.optim.AdamW(network.parameters(), lr=lr)
    if optimizer_method == 'WarmUp':
        dim = network.module.src_embed[0].d_model if torch.cuda.device_count() > 1 else network.src_embed[0].d_model 
        return NoamOpt(dim, factor=factor, warmup=warmup, optimizer=torch.optim.Adam(network.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), resume_from_step=resume_from_step) 


"""# loss function"""
def mae_for_normalized(preds, labels, original_scale_label, mask=False, null_val=np.nan):
    '''With normalized data, no labels will give out zero value, hence, original scale label is needed'''
    loss = torch.abs(preds - labels)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(original_scale_label)
        else:
            mask = (original_scale_label!=null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        # loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def build_loss(loss_type, weighted):
    if loss_type == 'MSE':
        loss_func = nn.MSELoss # CAUTION: MSELoss() by default is averaged over batch 
    if loss_type == 'MAE':
        loss_func = nn.L1Loss
    if loss_type == 'smoothl1':
        loss_func = nn.SmoothL1Loss
    if loss_type == 'mask_mae':
        return mae_for_normalized

    if weighted:
        loss_func = loss_func(reduction='none') # only calculate the squared error, no summation or average
    else:
        loss_func = loss_func() # CAUTION: MSELoss() by default is averaged over batch

    if loss_type == 'smoothl1':
        loss_func = nn.SmoothL1Loss(beta=0.1)
        print(f'loss function is {loss_type}, current beta is {0.1}, remember to set the correct beta!')
    
    print(f'loss function is {loss_type}, and it is {weighted} weighted')

    return loss_func

"""# Early Stopping""" 
# ref: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


# early_stopping = EarlyStopping(tolerance=2, min_delta=5) 