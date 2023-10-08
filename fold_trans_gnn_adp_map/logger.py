# writer and meta data
import math
import os
import torch
# plots
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import shutil
# sensor maps
import seaborn as sns
# save to json
from config import save_to_json
from torch.utils.tensorboard import SummaryWriter

""" # log dict update """
def initialize_lowest_log_dict(metric_key_list):
    lowest = {'loss':{'lowest':math.inf, 'epoch':0}}
    for name in metric_key_list:
        lowest[name] = {'lowest':math.inf, 'epoch':0}

    return lowest 

def update_lowest_log_dict(lowest_log_dict, log_dict, epoch, epsilon):
    for metric_name, metric_value in log_dict.items():
        if 'loss' in metric_name:
            if lowest_log_dict['loss']['lowest'] - metric_value > epsilon:
                lowest_log_dict['loss']['lowest'] = metric_value
                lowest_log_dict['loss']['epoch'] = epoch 
        else: 
            if lowest_log_dict[metric_name]['lowest'] - metric_value > epsilon:
                lowest_log_dict[metric_name]['lowest'] = metric_value
                lowest_log_dict[metric_name]['epoch'] = epoch

    return lowest_log_dict

""" # save best model """
def update_best_model(lowest_log_dict, log_dict, epoch, epsilon, model_save, network, total_epochs, Parallel):
    for metric_name, metric_value in log_dict.items():
        if 'loss' in metric_name:
            if lowest_log_dict['loss']['lowest'] - metric_value > epsilon:
                model_save(network, num_epoch=epoch, total_epochs=total_epochs, Parallel=Parallel, save_type='loss')
        else: 
            if lowest_log_dict[metric_name]['lowest'] - metric_value > epsilon:
                model_save(network, num_epoch=epoch, total_epochs=total_epochs, Parallel=Parallel, save_type=metric_name)

""" # save best beta """
def update_beta_index(lowest_log_dict, log_dict, epoch, epsilon, tensor_save, tensors, filename, total_epochs):
    for metric_name, metric_value in log_dict.items():
        if 'loss' in metric_name:
            if lowest_log_dict['loss']['lowest'] - metric_value > epsilon:
                tensor_save(tensors, filename, num_epoch=epoch, total_epochs=total_epochs, save_type='loss', )
        else: 
            if lowest_log_dict[metric_name]['lowest'] - metric_value > epsilon:
                tensor_save(tensors, filename, num_epoch=epoch, total_epochs=total_epochs, save_type=metric_name)

""" # writer """
def log_writer_batch(writer, log_dict, batch_id):
    for key, value in log_dict.items():
        if 'loss' in key:
            writer.add_scalar('loss', value, batch_id)
        else:
            writer.add_scalar(key, value, batch_id)

def log_writer_epoch(writer, log_dict, epoch):
    for key, value in log_dict.items():
        if 'loss' in key:
            writer.add_scalar('LOSS', value, epoch)
        else:
            writer.add_scalar(key.upper(), value, epoch)

def log_writer_hparams(writer, tunable_configs, log_dict):
    writer.add_hparams(tunable_configs.__dict__, log_dict)

def log_writer_histogram(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(name, param, epoch)

def writer_init(meta_save, log_batch=False, log_hp=False):
    train_writer = SummaryWriter(meta_save.path_log_train)
    val_writer = SummaryWriter(meta_save.path_log_val)
    graph_writer = SummaryWriter(meta_save.path_log_graph)
    
    if log_batch:
        batch_train_writer = SummaryWriter(meta_save.path_log_train_batch)
        batch_val_writer = SummaryWriter(meta_save.path_log_val_batch)
    else:
        batch_train_writer = None
        batch_val_writer = None

    if log_hp:
        hp_writer = SummaryWriter(meta_save.path_log_hp)
    else:
        hp_writer = None
    
    return train_writer, val_writer, graph_writer, batch_train_writer, batch_val_writer, hp_writer

""" #save model and meta data"""
class SaveMeta(object):
    def __init__(self, root_dir='./', model_name='DirRec_', process=0, log_batch=False, log_hp=False, hp_iter=None):
        
        self.path = self.makedir_log_model_time(root_dir, model_name, process)

        if hp_iter:
            self.path_log_train = self.save_log_path('train/exp'+str(hp_iter).zfill(2))
            self.path_log_val = self.save_log_path('val/exp'+str(hp_iter))
            self.path_log_graph = self.save_log_path('graph/exp'+str(hp_iter).zfill(2))

            if log_batch:
                self.path_log_train_batch = self.save_log_path('train_batch/exp'+str(hp_iter).zfill(2))
                self.path_log_val_batch = self.save_log_path('val_batch/exp'+str(hp_iter).zfill(2))
            
            if log_hp:
                self.path_log_hp = self.save_log_path('hp/exp'+str(hp_iter).zfill(2))

        else:
            self.path_log_train = self.save_log_path('train')
            self.path_log_val = self.save_log_path('val')
            self.path_log_graph = self.save_log_path('graph')

            if log_batch:
                self.path_log_train_batch = self.save_log_path('train_batch')
                self.path_log_val_batch = self.save_log_path('val_batch')
            
            if log_hp:
                self.path_log_hp = self.save_log_path('hp')

        
    # save log model
    def makedir_log_model_time(self, root_dir, model_name, process):
        path_dir = os.path.join(root_dir, model_name + str(process).zfill(3))
        for dir in ['model', 'runs', 'meta_data']:
            path = os.path.join(path_dir, dir)
            # print(path)
            os.makedirs(path, exist_ok=True)
        
        return path_dir

    # save model
    def save_model(self, network, num_epoch=0, total_epochs=1000, Parallel=True, save_type='regular'):
        filled_zeros = len(str(total_epochs))
        if save_type == 'regular':
            path_dir = os.path.join(self.path, 'model')
            path_file = 'net_' + str(num_epoch).zfill(filled_zeros) + '.pth'
        else:
            path_dir = os.path.join(self.path, 'model')
            path_file = 'best_' + save_type +'_model_' + '.pth'

        path = os.path.join(path_dir, path_file)
        # print(path)

        if Parallel:
            # https://www.zhihu.com/question/67726969
            torch.save(network.module.state_dict(), path)
        else:
            torch.save(network.state_dict(), path) 
    
    # save supports, mask and more
    def save_tensors(self, tensors, filename):
        """ notice that it does not have to be torch.tensor, could be list of tensors or others things"""
        path = os.path.join(self.path, 'meta_data', filename)
        torch.save(tensors, path)

    # save supports, mask and more
    def save_beta_index(self, tensors, filename, num_epoch=0, total_epochs=1000, save_type='regular'):
        """ save beta and index for best models"""
        filled_zeros = len(str(total_epochs))
        path_dir = os.path.join(self.path, 'model') 
        if save_type == 'regular':
            path_file = filename + '_' + str(num_epoch).zfill(filled_zeros) + '.pt'
        else:
            path_file = 'best_' + save_type +'_' + filename + '_' + '.pt'

        path = os.path.join(path_dir, path_file)
        torch.save(tensors, path)

    def save_log_path(self, log_type='train'):
        path = os.path.join(self.path,'runs', log_type)

        return path

    def output_metadata_train(self, dataset, batch_size_train, batch_num_train, batch_size_val, batch_num_val,
                              total_epochs, computation_time, lowest_train:dict, lowest_val:dict):

        path_dir = os.path.join(self.path, 'meta_data')

        output_dict = {}
        output_dict['dataset'] = dataset
        output_dict['batch_size_train'] = batch_size_train
        output_dict['batch_num_train'] = batch_num_train
        output_dict['batch_size_val'] = batch_size_val
        output_dict['batch_num_val'] = batch_num_val

        output_dict['total_epochs'] = total_epochs
        output_dict['duration_in_sec'] = computation_time

        output_dict['train_stats'] = lowest_train
        output_dict['val_stats'] = lowest_val
        
        save_to_json(output_dict, 'train_metrics.json', path_dir)
    
    
    def output_metadata_test(self, test_dict:dict, computation_time, batch_size, batch_num):
        path_dir = os.path.join(self.path, 'meta_data')

        output_dict = {}
        output_dict['duration_in_sec'] = computation_time
        output_dict['batch_size'] = batch_size
        output_dict['batch_num'] = batch_num
        output_dict['test_stats'] = test_dict

        save_to_json(output_dict, 'test_metrics.json', path_dir)

    def output_config(self, config, filename):
        path_dir = os.path.join(self.path, 'meta_data') 
        save_to_json(config, filename, path_dir)

class SavePlots(object):
    def __init__(self, root_dir='./', model_name='DirRec_', process=0):
        self.path = os.path.join(root_dir, model_name + str(process).zfill(3))
        self.plots_path = self.makedir_plots(self.path)

    # save plots
    def makedir_plots(self, root_path):
        path_dir = os.path.join(root_path, 'meta_data', 'test_plots')
        # if dir exists, then overwrite
        if os.path.exists(path_dir):
            shutil.rmtree(path_dir)
        os.makedirs(path_dir)

        return path_dir

    @staticmethod
    def tensor_to_numpy(inputs, targets, outputs):
        """ convert GPU tensor to numpy.array on cpu to plot

        all inputs are assumed to be in no_grad() by default, if used in training phase, be cautious

        Args:
            inputs: torch.tensor, shape [batch_size, feature, input_length]
            targets: torch.tensor, shape [batch_size, feature, target_length] 
            outputs: torch.tensor, shape [batch_size, feature, target_length]

        Returns:
            (dummy_x, observed): (index data for x axis, inputs+targets)
            (dummy_x_output, outputs): (index data for x axis, outputs) 
            (input_len, outputs_len): length for inputs and outputs, helper return for plotting

        """

        assert targets.shape == outputs.shape, "outputs and targets should have the same shape, check code"

        inputs = inputs.flatten().to('cpu').numpy()
        targets = targets.flatten().to('cpu').numpy()
        outputs = outputs.flatten().to('cpu').numpy()
        
        observed = np.concatenate((inputs, targets))
        full_len = observed.shape[0]
        input_len = inputs.shape[0]
        outputs_len = outputs.shape[0]
        dummy_x = np.arange(0, full_len)
        dummy_x_output = np.arange(input_len, full_len)

        return (dummy_x, observed), (dummy_x_output, outputs), (input_len, outputs_len)

    def save_plots(self, observed, forecasted, helper_len, loss, index):
        """ plot and save figure

        Args:
            observed: (dummy_x, observed) 
            forecasted: (dummy_x_output, outputs)
            helper_len: (input_len, outputs_len) 
            loss: loss for current sample, in value not tensor
            index: sample plotted indicator; this function should be used with dataloader batch_size 1
        """

        # set up subplot figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10), facecolor='w')
        ax.plot(observed[0], observed[1], marker='.', label='Observed')
        ax.plot(forecasted[0], forecasted[1], marker='^',label='Forecasted')
        ax.plot([], [], color='w', label=f'current loss: {loss:.3f}') # empty graph to show loss of current sample
        ax.axvline(helper_len[0], color='seagreen', linestyle='--') # shows the start of prediction
        ax.grid(axis='y', color='grey', ls='-.')

        # adjust xy ticks
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 0.2))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start+1, end+1, 1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        # set label, title, legend
        ax.set_xlabel('index', fontsize=18)
        ax.set_ylabel('value', fontsize=18)
        ax.set_title('Forecasted vs. Observed', fontsize=20)
        ax.legend(fontsize=16)

        # add annotation to outputs and targets
        for x, y_pred, y_true in zip(forecasted[0], forecasted[1], observed[1][-helper_len[1]:]):
            label_pred = "{:.3f}".format(y_pred)
            ax.annotate(label_pred, # this is the text
                        (x, y_pred), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,6), # distance from text to points (x,y)
                        ha='center', # horizontal alignment can be left, right or center
                        fontsize=14) 

            label_true = "{:.3f}".format(y_true) 
            ax.annotate(label_true, # this is the text
                        (x, y_true), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,-16), # distance from text to points (x,y)
                        ha='center', # horizontal alignment can be left, right or center
                        fontsize=14) 

        # save figure
        save_by_name = self.plots_path + '/' + 'forecasted_vs_observed_' + str(index).zfill(3) + '.png'
        fig.savefig(save_by_name, dpi=100, bbox_inches='tight')

        # don't output figure
        plt.close(fig)

# visualize sensor map
def visualize_sensor_map(network, n_stack, print_every_k_stack, supports_len, save_fig=False, root_dir='.'):
    """ plot sensor map, which is supports in transformer
    Args:
        network: network after testing
        n_stack: number of stacks/layers, equals to t_config.N
        print_every_k_stack: output every k stack
        supports_len: how many maps in supports list
        save_fig: save fig to some root dir
        root_dir: where to save figs
         
    """
    if supports_len == 1:
        # encoder
        for layer in range(0, n_stack, print_every_k_stack):
            fig, axn = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(network.encoder.gcn_layers[layer].supports[0].to('cpu'),
                        cbar=True,
                        vmin=0, vmax=1)      
            fig.suptitle('sensor map for encoder: layer {}'.format(str(layer)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'encoder_sensor_map_{}.png'.format(str(layer))))
            
    else:
        # decoder
        for layer in range(0, n_stack, print_every_k_stack):
            fig, axn = plt.subplots(1, supports_len, sharex=True, sharey=True, figsize=(20, 10))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            
            for i, ax in enumerate(axn.flat):
                ax.set_title('supports_'+str(i), fontsize=16) 
                sns.heatmap(network.decoder.gcn_layers[layer].supports[i].to('cpu'), ax=ax,
                            cbar=i == 0,
                            vmin=0, vmax=1,
                            cbar_ax=None if i else cbar_ax) 
            
            fig.suptitle('sensor map for decoder: layer {}'.format(str(layer)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'decoder_sensor_map_{}.png'.format(str(layer))))

# visualize_sensor_map(network, 2, 1, 1, True)
# visualize_sensor_map(network, 2, 1, 2, True)

def visualize_att_maps(network, n_stack, print_every_k_stack, num_heads, which_sensor, enc_vs_dec, save_fig=False, root_dir='.'):
    """ plot sensor map, which is supports in transformer
    Args:
        network: network after testing
        n_stack: number of stacks/layers, equals to t_config.N
        print_every_k_stack: output every k stack
        num_heads: heads in transformer, equals to t_config.h
        which_sensor: choose a sensor to look as, between [0, num_sensors], negative means counting backwards
        enc_vs_dec: plot attention map from with block: encoder, decoder_self, decoder_src
        save_fig: save fig to some root dir
        root_dir: where to save figs
         
    """
    if enc_vs_dec == 'encoder':
        for layer in range(0, n_stack, print_every_k_stack):
            fig, axn = plt.subplots(1, num_heads, sharex=True, sharey=True, figsize=(20, 10))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, ax in enumerate(axn.flat):
                ax.set_title('head_'+str(i), fontsize=16)
                sns.heatmap(network.encoder.layers[layer].self_attn.attn[0, which_sensor, i].to('cpu'), ax=ax,
                            cbar=i == 0,
                            vmin=0, vmax=1,
                            cbar_ax=None if i else cbar_ax)
            fig.suptitle('attention map for encoder: sensor {} layer {}'.format(str(which_sensor), str(layer)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'encoder_attention_{}.png'.format(str(layer))))

    elif enc_vs_dec == 'decoder_self':
        for layer in range(0, n_stack, print_every_k_stack):
            fig, axn = plt.subplots(1, num_heads, sharex=True, sharey=True, figsize=(20, 10))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, ax in enumerate(axn.flat):
                ax.set_title('head_'+str(i), fontsize=16) 
                sns.heatmap(network.decoder.layers[layer].self_attn.attn[0, which_sensor, i].to('cpu'), ax=ax,
                            cbar=i == 0,
                            vmin=0, vmax=1,
                            cbar_ax=None if i else cbar_ax) 
            fig.suptitle('attention map for decoder_self: sensor {} layer {}'.format(str(which_sensor), str(layer)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'decoder_self_attention_{}.png'.format(str(layer))))

    elif enc_vs_dec == 'decoder_src':
        for layer in range(0, n_stack, print_every_k_stack):
            fig, axn = plt.subplots(1, num_heads, sharex=True, sharey=True, figsize=(20, 10))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, ax in enumerate(axn.flat):
                ax.set_title('head_'+str(i), fontsize=16) 
                sns.heatmap(network.decoder.layers[layer].src_attn.attn[0, which_sensor, i].to('cpu'), ax=ax,
                            cbar=i == 0,
                            vmin=0, vmax=1,
                            cbar_ax=None if i else cbar_ax)
            fig.suptitle('attention map for decoder_src, sensor {} layer {}'.format(str(which_sensor), str(layer)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'decoder_src_attention_{}.png'.format(str(layer))))
    else:
        raise ValueError('please specify whether the attention map is from encoder, decoder_self or decoder_src')

# visualize_att_maps(network, 2, 1, 2, -1, 'encoder', True)
# visualize_att_maps(network, 2, 1, 2, -1, 'decoder_self', True)
# visualize_att_maps(network, 2, 1, 2, -1, 'decoder_src', True)

# plot calculated sensor map
# enc_path = '/nas/guosuhan/gwnet/logs/metr-la/gcn_transformer_008/meta_data/cumu_supp_enc.pt'
# dec_path = '/nas/guosuhan/gwnet/logs/metr-la/gcn_transformer_008/meta_data/cumu_supp_dec.pt'

# enc_supp = torch.load(enc_path)
# dec_supp = torch.load(dec_path)

# root_dir = 'heatmaps_008'
# sample_ids = [3, 17]

def plot_encoder_map_single(encoder_supp, sample_id, stack_id, supp_id, normal_range, save_fig, root_dir):
    """ plot single decoder sensor maps 
    encoder_map: list of lists, structure [sample] [stack] [maps] 
    sample_id: sample index, incrementing from 0
    stack_id: stack id, incrementing from 0
    supp_id: supports id, incrementing from 0
    normal_range: bool, if True, plot map with range(0, 1), otherwise with deducted range
    save_fig: save fig to some root dir
    root_dir: where to save figs

    Caution: in figure name and title, all index starts at 1, not 0
    """ 
    fig, _ = plt.subplots(1, 1, figsize=(10, 10))
    if normal_range:
        sns.heatmap(encoder_supp[sample_id][stack_id][supp_id].to('cpu'),
                    cbar=True,
                    vmin=0, vmax=1) 
    else:
        sns.heatmap(encoder_supp[sample_id][stack_id][supp_id].to('cpu'),
                    cbar=True) 
    fig.suptitle('sensor map for encoder: sample {} stack {} supp {}'.format(str(sample_id+1), str(stack_id+1), str(supp_id+1)) , fontsize=20)
    if save_fig:
        plt.savefig(os.path.join(root_dir, 'encoder_sensor_map_{}_{}_{}.png'.format(str(sample_id+1), str(stack_id+1), str(supp_id+1))))

# plot_encoder_map_single(enc_supp, 3, 0, 1, False, True, root_dir)
# plot_encoder_map_single(enc_supp, 3, 1, 1, False, True, root_dir)
# plot_encoder_map_single(enc_supp, 17, 0, 1, False, True, root_dir)
# plot_encoder_map_single(enc_supp, 17, 1, 1, False, True, root_dir)

def plot_encoder_map_multiple(encoder_supp, sample_ids, n_stack, supports_len, save_fig, root_dir):
    """ plot multiple decoder sensor maps 
    decoder_supp: list of lists, structure [sample] [output_len] [stack] [maps] 
    sample_ids: tuple or list, if tuple, (start_sample_id, end_sample_id); if list, list of ids
    n_stack: number of stacks, aka N
    supports_len: number of supports, original included if used
    save_fig: save fig to some root dir
    root_dir: where to save figs

    Caution: in figure name, all index starts at 1, not 0
             in sub-figure title, sample indicates the incrementing status, not the real sample id
    """

    if isinstance(sample_ids, tuple):
        start_sample_id = sample_ids[0]
        end_sample_id = sample_ids[1]
        sample_ids = list(range(start_sample_id, end_sample_id))

    n_sample = len(sample_ids)
    
    for i, sample_id in enumerate(sample_ids):
        for stack_id in range(n_stack):

            fig, axn = plt.subplots(1, supports_len, sharex=True, sharey=True, figsize=(10*supports_len, 10))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])

            for supp_id, ax in enumerate(axn.flat):
                ax.set_title('supports_'+str(supp_id), fontsize=16) 
                sns.heatmap(encoder_supp[sample_id][stack_id][supp_id].to('cpu'), ax=ax,
                            cbar=supp_id == 0,
                            vmin=0, vmax=1,
                            cbar_ax=None if supp_id else cbar_ax) 

            fig.suptitle('sensor map: sample {}/{} stack {}/{}'.format(str(i+1), str(n_sample), str(stack_id+1), str(n_stack)), fontsize=20)
            if save_fig:
                plt.savefig(os.path.join(root_dir, 'encoder_sensor_map_{}_{}_all.png'.format(str(sample_id+1), str(stack_id+1))))   

# plot_encoder_map_multiple(enc_supp, sample_ids, 2, 2, True, root_dir)

def plot_decoder_map_single(decoder_supp, sample_id, output_id, stack_id, supp_id, normal_range, save_fig, root_dir):
    """ plot single decoder sensor maps 
    decoder_supp: list of lists, structure [sample] [output_len] [stack] [maps] 
    sample_id: sample index, incrementing from 0
    output_id: output length index, incrementing from 0 
    stack_id: stack id, incrementing from 0
    supp_id: supports id, incrementing from 0
    normal_range: bool, if True, plot map with range(0, 1), otherwise with deducted range
    save_fig: save fig to some root dir
    root_dir: where to save figs

    Caution: in figure name and title, all index starts at 1, not 0
    """ 
    fig, _ = plt.subplots(1, 1, figsize=(10, 10))
    if normal_range:
        sns.heatmap(decoder_supp[sample_id][output_id][stack_id][supp_id].to('cpu'),
                    cbar=True,
                    vmin=0, vmax=1) 
    else:
        sns.heatmap(decoder_supp[sample_id][output_id][stack_id][supp_id].to('cpu'),
                    cbar=True) 
    fig.suptitle('sensor map for decoder: sample {} output {} stack {} supp {}'.format(str(sample_id+1), str(output_id+1), str(stack_id+1), str(supp_id+1)) , fontsize=20)
    # plt.close(fig) 
    if save_fig:
        plt.savefig(os.path.join(root_dir, 'decoder_sensor_map_{}_{}_{}_{}.png'.format(str(sample_id+1), str(output_id+1), str(stack_id+1), str(supp_id+1))))

# plot_decoder_map_single(dec_supp, 3, 2, 0, 1, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 3, 2, 0, 2, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 3, 2, 1, 1, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 3, 2, 1, 2, False, True, root_dir)

# plot_decoder_map_single(dec_supp, 17, 2, 0, 1, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 17, 2, 0, 2, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 17, 2, 1, 1, False, True, root_dir)
# plot_decoder_map_single(dec_supp, 17, 2, 1, 2, False, True, root_dir) 

def plot_decoder_map_multiple(decoder_supp, sample_ids, output_len, n_stack, supports_len, save_fig, root_dir):
    """ plot multiple decoder sensor maps 
    decoder_supp: list of lists, structure [sample] [output_len] [stack] [maps] 
    sample_ids: tuple or list, if tuple, (start_sample_id, end_sample_id); if list, list of ids
    output_len: output/target length, confirm the max_len is the same
    n_stack: number of stacks, aka N
    supports_len: number of supports, original included if used
    save_fig: save fig to some root dir
    root_dir: where to save figs

    Caution: in figure name, all index starts at 1, not 0
             in sub-figure title, sample indicates the incrementing status, not the real sample id
    """

    if isinstance(sample_ids, tuple):
        start_sample_id = sample_ids[0]
        end_sample_id = sample_ids[1]
        sample_ids = list(range(start_sample_id, end_sample_id))

    n_sample = len(sample_ids)
    
    for i, sample_id in enumerate(sample_ids):
        for output_id in range(output_len):
            for stack_id in range(n_stack):

                fig, axn = plt.subplots(1, supports_len, sharex=True, sharey=True, figsize=(10*supports_len, 10))
                cbar_ax = fig.add_axes([.91, .3, .03, .4])

                for supp_id, ax in enumerate(axn.flat):
                    ax.set_title('supports_'+str(supp_id), fontsize=16) 
                    sns.heatmap(decoder_supp[sample_id][output_id][stack_id][supp_id].to('cpu'), ax=ax,
                                cbar=supp_id == 0,
                                vmin=0, vmax=1,
                                cbar_ax=None if supp_id else cbar_ax) 

                fig.suptitle('sensor map: sample {}/{} output_len {}/{} stack {}/{}'.format(str(i+1), str(n_sample), str(output_id+1), str(output_len), str(stack_id+1), str(n_stack)), fontsize=20)
                if save_fig:
                    plt.savefig(os.path.join(root_dir, 'decoder_sensor_map_{}_{}_{}_all.png'.format(str(sample_id+1), str(output_id+1), str(stack_id+1))))   

# plot_decoder_map_multiple(dec_supp, sample_ids, 3, 2, 3, True, root_dir)