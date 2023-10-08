import collections
import copy
import torch
import time
import datetime

from utils import load_pickle, set_seed
from dataset import build_dataset, Batch
from GConv import build_network, count_parameters, batch_greedy_decode
from beta_controller import *
from metrics import build_metric
from loss_and_optimizer import build_optimizer_linear, build_loss, EarlyStopping
from config import configs, configs_all, save_to_json
from logger import(
    initialize_lowest_log_dict,
    update_lowest_log_dict,
    update_best_model,
    update_beta_index,
    log_writer_batch,
    log_writer_epoch,
    log_writer_hparams,
    writer_init,
    SaveMeta
)

def cumulative_elementwise_sum(ds):
    result = collections.Counter()
    for d in ds:
        result.update(d)
        yield dict(result)

def dict_values_division(mydict, denom):
  return {k: v / denom for k, v in mydict.items()}


def train_epoch(network, loader, loss_fn, scaler, opt_and_scheduler, optimizer_method, norm_method, target_dim, device, metric_key_list, batch_writer, beta, epoch_id, beta_range, updated_index, agreement_threshold):
    cumu_loss = 0
    cumu_metric = {}
    num_samples = len(loader)

    for sample_id, (inputs, targets, targets_raw) in enumerate(loader):
        # [batch_size, sensors, seq_len, dim]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # create mask using Batch class
        batch = Batch(inputs, targets, pad=-10, device=device) # did not implement padding, -10 is arbitrary

        # zero the parameter gradients
        if optimizer_method=='WarmUp':
            opt_and_scheduler.optimizer.zero_grad()
        else:
            opt_and_scheduler.zero_grad()
        
        # ➡ Forward pass
        # print(batch.src.shape)
        outputs, sf_enc = network(batch.src, batch.src_mask, beta)
        # print(outputs.shape)
        # # original loss
        loss = loss_fn(outputs, batch.trg_y)
        # masked loss
        # unmasked_loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=False, null_val=0.0) 
        # loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=True, null_val=0.0)
        # print(f'training loss {unmasked_loss}, masked loss {loss}')

        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        opt_and_scheduler.step()

        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)

        # beta update
        enc_beta = beta[:, :, 0].unsqueeze(-1) 
        # dec_beta = beta[:, :, 1:] 
        enc_updated_index = updated_index[:, 0].unsqueeze(-1)
        # dec_updated_index = updated_index[:, 1:]
        # print(f'sparsity_flags: {sf_enc}, {sf_dec}')
        enc_beta, enc_updated_index = beta_apply(enc_beta, sf_enc, beta_range, sample_id, num_samples, epoch_id, enc_updated_index, agreement_threshold, device)
        # dec_beta, dec_updated_index = beta_apply(dec_beta, sf_dec, beta_range, sample_id, num_samples, epoch_id, dec_updated_index, agreement_threshold, device)
        beta = enc_beta
        # print(f'updated index passed {updated_index}')
        updated_index = enc_updated_index
        # print(f'new updated index {updated_index}')

        # metrics
        with torch.no_grad():
            # outputs_raw = scaler.inverse_transform(outputs.to('cpu'), norm_method)
            if target_dim == 1:
                outputs_raw = scaler[0].inverse_transform(outputs.to('cpu'), norm_method)
            elif target_dim > 1 and target_dim <= len(scaler):
                outputs_raw = torch.cat([var.inverse_transform(outputs.to('cpu'), norm_method) for var in scaler], dim=-1) # normalize per dim and concat back on dim 
            else:
                raise ValueError('target dim should be smaller or equal to number of datasets/scalers')

            # metrics for zscored values
            # metrics = build_metric(outputs, targets, raw_labels=targets_raw, metric_key_list=metric_key_list) 
            # metrics for raw values
            # print(outputs_raw.shape, targets_raw.shape)
            metrics = build_metric(outputs_raw, targets_raw, metric_key_list=metric_key_list)

        if not cumu_metric: # if cumu_metric is an empty dict
            cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
        else:
            cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]

        if batch_writer:
            loss_dict = {"batch_train_loss": loss.item()}
            metrics.update(loss_dict)
            log_writer_batch(batch_writer, metrics, sample_id)

    return cumu_loss / len(loader), dict_values_division(cumu_metric, len(loader)), beta, updated_index

def dev_epoch(network, loader, loss_fn, scaler, norm_method, target_dim, device, metric_key_list, batch_writer, beta):
    cumu_loss = 0
    cumu_metric = {}

    with torch.no_grad():
        for i, (inputs, targets, targets_raw) in enumerate(loader):
            # [batch_size, sensors, seq_len, dim]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # create mask using Batch class
            batch = Batch(inputs, targets, pad=-10, device=device) # did not implement padding, -10 is arbitrary
            
            # ➡ Forward pass only
            outputs, _ = network(batch.src, batch.src_mask, beta)
            # # original loss
            loss = loss_fn(outputs, batch.trg_y)
            # masked loss
            # unmasked_loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=False, null_val=0.)
            # loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=True, null_val=0.)
            # print(f'validation loss {unmasked_loss}, masked loss {loss}')

            cumu_loss += loss.item()

            # metrics
            with torch.no_grad():
                # outputs_raw = scaler.inverse_transform(outputs.to('cpu'), norm_method)
                if target_dim == 1:
                    outputs_raw = scaler[0].inverse_transform(outputs.to('cpu'), norm_method)
                elif target_dim > 1 and target_dim <= len(scaler):
                    outputs_raw = torch.cat([var.inverse_transform(outputs.to('cpu'), norm_method) for var in scaler], dim=-1) # normalize per dim and concat back on dim 
                else:
                    raise ValueError('target dim should be smaller or equal to number of datasets/scalers')

            # metrics for zscored values
            # metrics = build_metric(outputs, targets, raw_labels=targets_raw, metric_key_list=metric_key_list) 
            # metrics for raw values
            metrics = build_metric(outputs_raw, targets_raw, metric_key_list=metric_key_list)

            if not cumu_metric: # if cumu_metric is an empty dict
                cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
            else:
                cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]     

            if batch_writer:
                loss_dict = {"batch_val_loss": loss.item()}
                metrics.update(loss_dict)
                log_writer_batch(batch_writer, metrics, i)

    return cumu_loss / len(loader), dict_values_division(cumu_metric, len(loader))

def train(config=None, tunable_config=None, device='cpu', supports=None, beta=None, updated_index=None,
          train_loader=None, val_loader=None, scaler=None,
          train_writer=None, val_writer=None, graph_writer=None, hp_writer=None,
          batch_train_writer=None, batch_val_writer=None,
          model_save=None, beta_save=None, tensors_save=None, early_stopping=None):

    # if config.naive_vs_geom == 'naive':
    #     train_loader, val_loader, _, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
    #                                                         config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
    #                                                         pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, gamble=config.gamble, all_random=config.all_random, exclude_features=config.exclude_features) 
    # elif config.naive_vs_geom == 'geom':
    #     train_loader, val_loader, _, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
    #                                                         config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
    #                                                         pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, lm=config.lm, mode=config.mode, distribution=config.distribution, exclude_sensors=config.exclude_sensors, all_random=config.all_random, exclude_features=config.exclude_features)
    # else:
    #     raise NotImplementedError

    network = build_network(config.src_seq_len, config.trg_seq_len, config.input_dim, config.target_dim,
                            tunable_config.N, tunable_config.d_model, tunable_config.d_ff, tunable_config.h, tunable_config.dropout, 
                            tunable_config.add_gcn, config.sensors, supports,
                            config.set_diag, config.undirected, config.truncate, config.threshold, config.sparsity_ratio, config.num_layers, config.bn, config.conv_type, config.num_maps, config.adp_supp_len,
                            device, True, config.load_net_path)
    print(f'current network has {count_parameters(network)} parameters')
    opt_and_scheduler = build_optimizer_linear(network, config.lr, t_config.optimizer_method, config.factor, config.warmup, config.resume_from_step)
    loss_fn = build_loss(config.loss_type, False)

    # initiate log to document the lowest loss/metrics
    lowest_train = initialize_lowest_log_dict(config.metric_key_list)
    lowest_val = initialize_lowest_log_dict(config.metric_key_list)

    train_beta = beta
    for epoch in range(config.epochs):
        network.train()
        b_range = beta_range(config.beta_scale, config.beta_upper_lim, len(train_loader), config.epochs)
        avg_loss_train, avg_metrics_train, train_beta, updated_index = train_epoch(network, train_loader, loss_fn, scaler, opt_and_scheduler, t_config.optimizer_method, config.norm_method, config.target_dim, device, config.metric_key_list, batch_train_writer, train_beta, epoch, b_range, updated_index, config.agreement_threshold)
        loss_dict_train = {"train_loss": avg_loss_train}
        avg_metrics_train.update(loss_dict_train)
        if train_writer:
            log_writer_epoch(train_writer, avg_metrics_train, epoch)

        network.eval()
        avg_loss_val, avg_metrics_val = dev_epoch(network, val_loader, loss_fn, scaler, config.norm_method, config.target_dim, device, config.metric_key_list, batch_val_writer, train_beta) # should not change beta during validation
        loss_dict_val = {"val_loss": avg_loss_val}
        avg_metrics_val.update(loss_dict_val)
        if val_writer:
            log_writer_epoch(val_writer, avg_metrics_val, epoch)

        string = 'epoch {}:\ntrain: loss: {:.4f} mae: {:.4f} mask_mae: {:.4f} mse: {:.4f} mask_mse: {:.4f} rmse: {:.4f} mask_rmse: {:.4f} mape: {:.4f} clip_mape: {:.4f} mask_mape: {:.4f} smape: {:.4f}\nvalid: loss: {:.4f} mae: {:.4f} mask_mae: {:.4f} mse: {:.4f} mask_mse: {:.4f} rmse: {:.4f} mask_rmse: {:.4f} mape: {:.4f} clip_mape: {:.4f} mask_mape: {:.4f} smape: {:.4f}' 
        print(string.format(epoch, 
                            avg_metrics_train['train_loss'], avg_metrics_train['mae'], avg_metrics_train['mae_mask'], avg_metrics_train['mse'], avg_metrics_train['mse_mask'], avg_metrics_train['rmse'], avg_metrics_train['rmse_mask'], avg_metrics_train['mape'], avg_metrics_train['mape_clip'], avg_metrics_train['mape_mask'], avg_metrics_train['smape'],  
                            avg_metrics_val['val_loss'], avg_metrics_val['mae'], avg_metrics_val['mae_mask'], avg_metrics_val['mse'], avg_metrics_val['mse_mask'], avg_metrics_val['rmse'], avg_metrics_val['rmse_mask'], avg_metrics_val['mape'], avg_metrics_val['mape_clip'], avg_metrics_val['mape_mask'], avg_metrics_val['smape']))
        print(f'current beta is {train_beta}')

        # save current beta
        update_beta_index(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps, beta_save, train_beta, 'beta', config.epochs)
        update_beta_index(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps, beta_save, updated_index, 'updated_index', config.epochs)

        # save best models
        update_best_model(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps, model_save, network, config.epochs, Parallel=True if torch.cuda.device_count() > 1 else False)

        # update dict
        lowest_train = update_lowest_log_dict(lowest_train, avg_metrics_train, epoch, config.log_lowest_eps)
        lowest_val = update_lowest_log_dict(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps)

        # save every 5 epochs
        if epoch % config.save_every == config.save_every-1 and model_save:
            model_save(network, num_epoch=epoch, total_epochs=config.epochs, Parallel=True if torch.cuda.device_count() > 1 else False, save_type='regular')
            # True if torch.cuda.device_count() > 1 else False
            beta_save(train_beta, 'beta', num_epoch=epoch, total_epochs=config.epochs, save_type='regular')
            beta_save(updated_index, 'updated_index', num_epoch=epoch, total_epochs=config.epochs, save_type='regular')
        
        # early stopping
        if early_stopping:
            early_stopping(avg_metrics_train['train_loss'], avg_metrics_val['val_loss'])
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
    
    if graph_writer:
        dummy_inputs = torch.ones(config.batch_size[0], config.sensors, config.src_seq_len, config.input_dim).to(device)
        dummy_targets = torch.ones(config.batch_size[0], config.sensors, config.trg_seq_len, config.target_dim).to(device)
        batch = Batch(dummy_inputs, dummy_targets, pad=-10, device=device)

        if torch.cuda.device_count() > 1:
            graph_writer.add_graph(network.module, [batch.src, batch.trg, batch.src_mask, batch.trg_mask, beta])
        else:
            graph_writer.add_graph(network, [batch.src, batch.trg, batch.src_mask, batch.trg_mask, beta])
        graph_writer.close()
    
    if hp_writer:
        log_writer_hparams(hp_writer, tunable_config, avg_metrics_val)
    
    return lowest_train, lowest_val, train_loader.batch_size, val_loader.batch_size, len(train_loader), len(val_loader)


def test(config=None, tunable_config=None, test_loader=None, scaler=None, supports=None, beta=None, device='cpu'):
    # # notice that test data do not pollute
    # _, _, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,  
    #                                           config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
    #                                           pollute=False)
    network = build_network(config.src_seq_len, config.trg_seq_len, config.input_dim, config.target_dim,
                            tunable_config.N, tunable_config.d_model, tunable_config.d_ff, tunable_config.h, tunable_config.dropout, 
                            tunable_config.add_gcn, config.sensors, supports,
                            config.set_diag, config.undirected, config.truncate, config.threshold, config.sparsity_ratio, config.num_layers, config.bn, config.conv_type, config.num_maps, config.adp_supp_len,
                            device, False, config.load_net_path)

    network.eval()
    print(f'current network has {count_parameters(network)} parameters')
    loss_fn = build_loss(config.loss_type, False)

    cumu_loss = 0
    cumu_metric = {}
    cumu_supp_enc = []
    cumu_supp_dec = []

    out_list = []

    with torch.no_grad():
        for _, (inputs, targets, targets_raw) in enumerate(test_loader):
            # [batch_size, sensors, seq_len, dim]
            inputs = inputs.to(device)
            targets = targets.to(device)

            # create mask using Batch class
            batch = Batch(inputs, targets, pad=-10, device=device) # did not implement padding, -10 is arbitrary
            
            # ➡ Forward pass only
            outputs, supp_enc = network(batch.src, batch.src_mask, beta)
            cumu_supp_enc.append(supp_enc)
            # cumu_supp_dec.append(supp_dec)
            # # original loss
            loss = loss_fn(outputs, batch.trg_y)
            # masked loss
            # unmasked_loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=False, null_val=0.)
            # loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=True, null_val=0.)
            # print(f'test loss {unmasked_loss}, masked loss {loss}')

            cumu_loss += loss.item()

            # metrics
            # outputs_raw = scaler.inverse_transform(outputs.to('cpu'), config.norm_method)
            if config.target_dim == 1:
                outputs_raw = scaler[0].inverse_transform(outputs.to('cpu'), config.norm_method)
            elif config.target_dim > 1 and config.target_dim <= len(scaler):
                outputs_raw = torch.cat([var.inverse_transform(outputs.to('cpu'), config.norm_method) for var in scaler], dim=-1) # normalize per dim and concat back on dim 
            else:
                raise ValueError('target dim should be smaller or equal to number of datasets/scalers')
            
            # metrics for zscored values
            # metrics = build_metric(outputs, targets, raw_labels=targets_raw, metric_key_list=metric_key_list) 
            # metrics for raw values
            out_list.append(outputs_raw)
            metrics = build_metric(outputs_raw, targets_raw, metric_key_list=config.metric_key_list)

            if not cumu_metric: # if cumu_metric is an empty dict
                cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
            else:
                cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]

    loss_dict = {'test_loss': cumu_loss / len(test_loader)}
    test_dict = dict_values_division(cumu_metric, len(test_loader)) 
    test_dict.update(loss_dict)

    return out_list, test_dict, test_loader.batch_size, len(test_loader), cumu_supp_enc, cumu_supp_dec



def generate_loader(config=None, tensors_save=None, fold=0):
    if config.split_by_fold:
        # print('correct track')
        if config.naive_vs_geom == 'naive':
            train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
                                                                fold=fold, num_folds=config.num_folds, num_test_samples=config.num_test_samples,
                                                                pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, gamble=config.gamble, all_random=config.all_random, exclude_features=config.exclude_features) 
        elif config.naive_vs_geom == 'geom':
            train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
                                                                fold=fold, num_folds=config.num_folds, num_test_samples=config.num_test_samples, 
                                                                pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, lm=config.lm, mode=config.mode, distribution=config.distribution, exclude_sensors=config.exclude_sensors, all_random=config.all_random, exclude_features=config.exclude_features)
        else:
            raise NotImplementedError
    else:
        if config.naive_vs_geom == 'naive':
            train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last,
                                                                fold=None, num_folds=config.num_folds, num_test_samples=config.num_test_samples,
                                                                pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, gamble=config.gamble, all_random=config.all_random, exclude_features=config.exclude_features) 
        elif config.naive_vs_geom == 'geom':
            train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
                                                                fold=None, num_folds=config.num_folds, num_test_samples=config.num_test_samples,
                                                                pollute=config.pollute, seq_len=config.src_seq_len, sensors=config.sensors, masking_ratio=config.masking_ratio, save_tensor=tensors_save, lm=config.lm, mode=config.mode, distribution=config.distribution, exclude_sensors=config.exclude_sensors, all_random=config.all_random, exclude_features=config.exclude_features)
        else:
            raise NotImplementedError
        
    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # set seed
    set_seed(17)

    # init configs
    config, t_config = configs([0, 0, 0, 0, 0, 2, 0])
    # t_config.d_ff = 64
    # t_config.d_model = 32

    # init supports
    supports = None
    if t_config.add_gcn and config.num_maps != config.adp_supp_len:
        adj_mx = load_pickle(config.pkl_filename)
        supports = torch.from_numpy(adj_mx).to(device)
    # supports = [torch.randn(137, 137).to('cpu')]
    # sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(config.pkl_filename)
    # supports = torch.from_numpy(adj_mx).to(device)
    # supports = None

    # init early_stopping
    if config.early_stopping and config.train_mode:
        early_stopping = EarlyStopping(tolerance=config.tolerance, min_delta=config.min_delta)
    else:
        early_stopping = None
        
    # init save_meta 
    meta_save = SaveMeta(root_dir=config.root_dir, model_name=config.model_name, process=config.process, 
                        log_batch=config.log_batch, log_hp=config.log_hp)

    # init beta
    if sum(config.adp_supp_len) > 0: # use self-learned maps
        beta_nums = sum(config.adp_supp_len)
    else: # only use distance matrix or do not use maps at all 
        beta_nums = 2
        
    num_gpus = torch.cuda.device_count()
    if config.train_mode:
        if config.load_net_path: # polluting stage
            beta = init_beta(beta_nums, t_config.N, num_gpus, 1., device)
        else: # finetune stage
            beta = init_beta(beta_nums, t_config.N, num_gpus, 1., device)
    else: # test
        # num_gpus = 1 # greedy decode only use 1 gpu
        beta = beta_load(config.load_net_path, 'beta')
        print(f'loaded beta \n{beta}')
        # beta = init_beta(beta_nums, t_config.N, num_gpus, 1., device)
        beta = beta[:num_gpus, ...].to(device)

    # init updated_index
    if config.train_mode:
        num_gpus = torch.cuda.device_count()
        updated_index = init_updated_index(beta_nums, t_config.N, device)
    
    if config.train_mode:
        meta_save.output_config(config, 'config.json')
        meta_save.output_config(t_config, 't_config.json')

        # init writers
        train_writer, val_writer, graph_writer, batch_train_writer, batch_val_writer, hp_writer = writer_init(meta_save, config.log_batch, config.log_hp)
        # train_writer = None
        # val_writer = None
        graph_writer = None
        # hp_writer = None
        # batch_train_writer = None
        # batch_val_writer = None


        # train
        training_time_0 = time.time()
        train_loader, val_loader, test_loader, scaler = generate_loader(config, meta_save.save_tensors, fold=config.current_fold)
        lowest_train, lowest_val, batch_size_train, batch_size_val, batch_num_train, batch_num_val = train(config=config, tunable_config=t_config, device=device, supports=supports, beta=beta, updated_index=updated_index,
                                                                                                            train_loader=train_loader, val_loader=val_loader, scaler=scaler, 
                                                                                                            train_writer=train_writer, val_writer=val_writer, graph_writer=graph_writer, hp_writer=hp_writer,
                                                                                                            batch_train_writer=batch_train_writer, batch_val_writer=batch_val_writer,
                                                                                                            model_save=meta_save.save_model,  beta_save=meta_save.save_beta_index, tensors_save=meta_save.save_tensors,
                                                                                                            early_stopping=early_stopping)
        training_time = time.time() - training_time_0
        # print(training_time) # seconds
        print(str(datetime.timedelta(seconds=training_time))) # days, hours:minutes:seconds

        meta_save.output_metadata_train(dataset=config.dataset_path, 
                                        batch_size_train=batch_size_train, batch_num_train=batch_num_train,
                                        batch_size_val=batch_size_val, batch_num_val=batch_num_val,
                                        total_epochs=config.epochs, computation_time=training_time, 
                                        lowest_train=lowest_train, lowest_val=lowest_val)
    else:
        # test
        testing_time_0 = time.time()
        train_loader, val_loader, test_loader, scaler = generate_loader(config, meta_save.save_tensors, fold=config.current_fold) 
        out_list, test_dict, batch_size_test, batch_num_test, cumu_supp_enc, cumu_supp_dec = test(config=config, tunable_config=t_config, test_loader=test_loader, scaler=scaler, supports=supports, beta=beta, device=device)
        testing_time = time.time() - testing_time_0
        # print(testing_time) # seconds
        print(str(datetime.timedelta(seconds=testing_time))) # days, hours:minutes:seconds

        meta_save.output_metadata_test(test_dict=test_dict, computation_time=testing_time, 
                                    batch_size=batch_size_test, batch_num=batch_num_test)
        
        meta_save.save_tensors(out_list, 'outputs_raw.pt')
        meta_save.save_tensors(cumu_supp_enc, 'cumu_supp_enc.pt')
        meta_save.save_tensors(cumu_supp_dec, 'cumu_supp_dec.pt')

