import torch
import torch.nn as nn
from metrics import mae, mae_m
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


# # resume training 2.0
# # intended
# # before train
# train_epochs = 20
# num_batches = 8 # 188
# model_size = 32
# # before ft
# best_train_epoch = 3
# factor = 2.5
# warmup = 20000

# print(f'total steps are {train_epochs*num_batches}')

def before_finetune(best_train_epoch, num_batches, factor, warmup, model_size):
    ''' decide best step and finetune epochs
    best_train_epoch: epoch of train to resume from
    num_batches: number of batches in each epoch 
    '''
    step = int(best_train_epoch * num_batches)
    lr_at_pt = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))) 
    print(f'best step {step}, lr {lr_at_pt}')
    return step, lr_at_pt

# resume_step, lr_ar_pt = before_finetune(best_train_epoch, num_batches, factor, warmup, model_size)

# # plot
def plot_lr(model_size, peak, factor, num_batches, train_epochs):
    num_steps = int(num_batches * train_epochs)
    opts = [NoamOpt(model_size, factor, peak[0], None, 0), 
            NoamOpt(model_size, factor, peak[1], None, 0),
            NoamOpt(model_size, factor, peak[2], None, 0)]
    plt.plot(np.arange(1, num_steps), [[opt.rate(i) for opt in opts] for i in range(1, num_steps)])
    plt.legend([f"{model_size}:{peak[0]}", f"{model_size}:{peak[1]}", f"{model_size}:{peak[2]}"])


# factor = 0.05
# peak = [50, 80, 100] 
# plot_lr(model_size, peak, factor, num_batches, train_epochs)

'''batch_size=32, d_model=32, num_batches=8, train_epochs=20
lr ~ 0.001
factor = 0.04, warmup = 50, lr ~ 0.001
factor = 0.05, warmup = 80, lr ~ 0.001
factor = 0.06, warmup = 100, lr~0.001
'''

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

def build_loss(loss_type, weighted):
    if loss_type == 'MSE':
        loss_func = nn.MSELoss # CAUTION: MSELoss() by default is averaged over batch 
    if loss_type == 'MAE':
        loss_func = nn.L1Loss
    if loss_type == 'smoothl1':
        loss_func = nn.SmoothL1Loss
    if loss_type == 'mask_mae':
        return mae

    if weighted:
        loss_func = loss_func(reduction='none') # only calculate the squared error, no summation or average
    else:
        loss_func = loss_func() # CAUTION: MSELoss() by default is averaged over batch
    
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

 