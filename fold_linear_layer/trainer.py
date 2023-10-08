import collections
import copy
import torch
import time
import datetime
import os
# from operator import itemgetter

from utils import load_pickle, set_seed
from dataset import build_dataset, Batch
from linear_layers import Model, DLModel, NLModel, count_parameters, train_vs_test
from metrics import build_metric
from loss_and_optimizer import build_optimizer_linear, build_loss, EarlyStopping, adjust_learning_rate
from config import configs, configs_all, save_to_json
from logger import(
    initialize_lowest_log_dict,
    update_lowest_log_dict,
    update_best_model,
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


def train_epoch(network, loader, loss_fn, scaler, opt_and_scheduler, optimizer_method, norm_method, target_dim, device, metric_key_list, batch_writer):
    cumu_loss = 0
    cumu_metric = {}

    for sample_id, (inputs, targets, targets_raw) in enumerate(loader):
        # [batch_size, sensors, seq_len, dim]
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch = Batch(inputs, targets, pad=-10, device=device)

        # zero the parameter gradients
        if optimizer_method=='WarmUp':
            opt_and_scheduler.optimizer.zero_grad()
        else:
            opt_and_scheduler.zero_grad()
        
        # ➡ Forward pass
        outputs = network(batch.src.permute(0, 2, 1, 3)) # [batch_size, seq_len, sensors, dim] 
        outputs = outputs.permute(0, 2, 1, 3) # [batch_size, seq_len, sensors, dim] -> [batch_size, sensors, seq_len, dim] 
        # original loss
        loss = loss_fn(outputs, batch.trg_y)
        # # masked loss
        # unmasked_loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=False, null_val=0.0) 
        # loss = loss_fn(outputs, batch.trg_y, targets_raw.to(device), mask=True, null_val=0.0)
        # print(f'training loss {unmasked_loss}, masked loss {loss}')

        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        opt_and_scheduler.step()

        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)

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
            metrics = build_metric(outputs_raw[..., :1], targets_raw[..., :1], metric_key_list=metric_key_list) # compute metric for incidence only
            # metrics = build_metric(outputs_raw, targets_raw, metric_key_list=metric_key_list)

        if not cumu_metric: # if cumu_metric is an empty dict
            cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
        else:
            cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]

        if batch_writer:
            loss_dict = {"batch_train_loss": loss.item()}
            metrics.update(loss_dict)
            log_writer_batch(batch_writer, metrics, sample_id)

    return cumu_loss / len(loader), dict_values_division(cumu_metric, len(loader))

def dev_epoch(network, loader, loss_fn, scaler, norm_method, target_dim, device, metric_key_list, batch_writer):
    cumu_loss = 0
    cumu_metric = {}

    with torch.no_grad():
        for i, (inputs, targets, targets_raw) in enumerate(loader):
            # [batch_size, sensors, seq_len, dim]
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch = Batch(inputs, targets, pad=-10, device=device)

            # ➡ Forward pass only
            outputs = network(batch.src.permute(0, 2, 1, 3)) # [batch_size, seq_len, sensors, dim] 
            outputs = outputs.permute(0, 2, 1, 3) # [batch_size, seq_len, sensors, dim] -> [batch_size, sensors, seq_len, dim] 
            # original loss
            loss = loss_fn(outputs, batch.trg_y)
            # # masked loss
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
            metrics = build_metric(outputs_raw[..., :1], targets_raw[..., :1], metric_key_list=metric_key_list) # compute metric for incidence only
            # metrics = build_metric(outputs_raw, targets_raw, metric_key_list=metric_key_list)

            if not cumu_metric: # if cumu_metric is an empty dict
                cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
            else:
                cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]     

            if batch_writer:
                loss_dict = {"batch_val_loss": loss.item()}
                metrics.update(loss_dict)
                log_writer_batch(batch_writer, metrics, i)

    return cumu_loss / len(loader), dict_values_division(cumu_metric, len(loader))

def train(config=None, tunable_config=None, device='cpu',
          train_loader=None, val_loader=None, scaler=None,
          train_writer=None, val_writer=None, graph_writer=None, hp_writer=None,
          batch_train_writer=None, batch_val_writer=None,
          model_save=None, tensors_save=None, early_stopping=None):

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

    if config.model_type == 'Linear':
        network = Model(config)
    elif config.model_type == 'DLinear': 
        network = DLModel(config)
    elif config.model_type == 'NLinear':
        network = NLModel(config)
    else:
        raise NotImplementedError
    
    network = train_vs_test(network, None, device, True)
    print(f'current network has {count_parameters(network)} parameters')
    opt_and_scheduler = build_optimizer_linear(network, config.lr, t_config.optimizer_method, config.factor, config.warmup, config.resume_from_step)
    loss_fn = build_loss(config.loss_type, False)

    # initiate log to document the lowest loss/metrics
    lowest_train = initialize_lowest_log_dict(config.metric_key_list)
    lowest_val = initialize_lowest_log_dict(config.metric_key_list)

    for epoch in range(config.epochs):
        network.train()
        avg_loss_train, avg_metrics_train = train_epoch(network, train_loader, loss_fn, scaler, opt_and_scheduler, t_config.optimizer_method, config.norm_method, config.target_dim, device, config.metric_key_list, batch_train_writer)
        loss_dict_train = {"train_loss": avg_loss_train}
        avg_metrics_train.update(loss_dict_train)
        if train_writer:
            log_writer_epoch(train_writer, avg_metrics_train, epoch)

        network.eval()
        avg_loss_val, avg_metrics_val = dev_epoch(network, val_loader, loss_fn, scaler, config.norm_method, config.target_dim, device, config.metric_key_list, batch_val_writer)
        loss_dict_val = {"val_loss": avg_loss_val}
        avg_metrics_val.update(loss_dict_val)
        if val_writer:
            log_writer_epoch(val_writer, avg_metrics_val, epoch)

        string = 'epoch {}:\ntrain: loss: {:.4f} mae: {:.4f} mask_mae: {:.4f} mse: {:.4f} mask_mse: {:.4f} rmse: {:.4f} mask_rmse: {:.4f} mape: {:.4f} clip_mape: {:.4f} mask_mape: {:.4f} smape: {:.4f}\nvalid: loss: {:.4f} mae: {:.4f} mask_mae: {:.4f} mse: {:.4f} mask_mse: {:.4f} rmse: {:.4f} mask_rmse: {:.4f} mape: {:.4f} clip_mape: {:.4f} mask_mape: {:.4f} smape: {:.4f}' 
        print(string.format(epoch, 
                            avg_metrics_train['train_loss'], avg_metrics_train['mae'], avg_metrics_train['mae_mask'], avg_metrics_train['mse'], avg_metrics_train['mse_mask'], avg_metrics_train['rmse'], avg_metrics_train['rmse_mask'], avg_metrics_train['mape'], avg_metrics_train['mape_clip'], avg_metrics_train['mape_mask'], avg_metrics_train['smape'],  
                            avg_metrics_val['val_loss'], avg_metrics_val['mae'], avg_metrics_val['mae_mask'], avg_metrics_val['mse'], avg_metrics_val['mse_mask'], avg_metrics_val['rmse'], avg_metrics_val['rmse_mask'], avg_metrics_val['mape'], avg_metrics_val['mape_clip'], avg_metrics_val['mape_mask'], avg_metrics_val['smape']))

        # save best models
        update_best_model(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps, model_save, network, config.epochs, Parallel=True if torch.cuda.device_count() > 1 else False)

        # update dict
        lowest_train = update_lowest_log_dict(lowest_train, avg_metrics_train, epoch, config.log_lowest_eps)
        lowest_val = update_lowest_log_dict(lowest_val, avg_metrics_val, epoch, config.log_lowest_eps)

        # save every 5 epochs
        if epoch % config.save_every == config.save_every-1 and model_save:
            model_save(network, num_epoch=epoch, total_epochs=config.epochs, Parallel=True if torch.cuda.device_count() > 1 else False, save_type='regular')
            # True if torch.cuda.device_count() > 1 else False
            
        # early stopping
        if early_stopping:
            early_stopping(avg_metrics_train['train_loss'], avg_metrics_val['val_loss'])
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
        
        # adjust learning rate
        adjust_learning_rate(opt_and_scheduler, epoch + 1, config)
    
    if graph_writer:
        dummy_inputs = torch.ones(config.batch_size[0], config.sensors, config.src_seq_len, config.input_dim).to(device)        

        if torch.cuda.device_count() > 1:
            graph_writer.add_graph(network.module, dummy_inputs)
        else:
            graph_writer.add_graph(network, dummy_inputs)
        graph_writer.close()
    
    if hp_writer:
        log_writer_hparams(hp_writer, tunable_config, avg_metrics_val)
    
    return lowest_train, lowest_val, train_loader.batch_size, val_loader.batch_size, len(train_loader), len(val_loader)
    

def test(config=None, test_loader=None, scaler=None, device='cpu'):
    # notice that test data do not pollute
    # _, _, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,  
    #                                           config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
    #                                           pollute=False)

    if config.model_type == 'linear':
        network = Model(config)
    elif config.model_type == 'DLinear': 
        network = DLModel(config)
    elif config.model_type == 'NLinear':
        network = NLModel(config)
    else:
        raise NotImplementedError
    # testing
    network = train_vs_test(network, config.load_net_path, device, False)
    print(f'current network has {count_parameters(network)} parameters')
    loss_fn = build_loss(config.loss_type, False)

    cumu_loss = 0
    cumu_metric = {}

    with torch.no_grad():
        output_ls = []
        output_raw_ls = []
        for _, (inputs, targets, targets_raw) in enumerate(test_loader):
            # [batch_size, sensors, seq_len, dim]
            inputs = inputs.to(device)
            targets = targets.to(device)


            batch = Batch(inputs, targets, pad=-10, device=device) 

            # ➡ Forward pass only
            outputs = network(batch.src.permute(0, 2, 1, 3)) # [batch_size, seq_len, sensors, dim] 
            outputs = outputs.permute(0, 2, 1, 3) # [batch_size, seq_len, sensors, dim] -> [batch_size, sensors, seq_len, dim] 

            # original loss
            loss = loss_fn(outputs, batch.trg_y)
            # # masked loss
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
            metrics = build_metric(outputs_raw[..., :1], targets_raw[..., :1], metric_key_list=config.metric_key_list) # compute metric for incidence only
            # metrics = build_metric(outputs_raw, targets_raw, metric_key_list=config.metric_key_list)

            if not cumu_metric: # if cumu_metric is an empty dict
                cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
            else:
                cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]
            
            # save outputs for plot
            output_ls.append(outputs)
            output_raw_ls.append(outputs_raw)

    loss_dict = {'test_loss': cumu_loss / len(test_loader)}
    test_dict = dict_values_division(cumu_metric, len(test_loader)) 
    test_dict.update(loss_dict)

    return test_dict, test_loader.batch_size, len(test_loader), output_ls, output_raw_ls

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

def main(config, t_config):
    # threads
    torch.set_num_threads(8)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # set seed
    set_seed(17)

    # init configs
    # config, t_config = configs([0])

    # init early_stopping
    if config.early_stopping and config.train_mode:
        early_stopping = EarlyStopping(tolerance=config.tolerance, min_delta=config.min_delta)
    else:
        early_stopping = None
        
    # init save_meta 
    meta_save = SaveMeta(root_dir=config.root_dir, model_name=config.model_name, process=config.process, 
                        log_batch=config.log_batch, log_hp=config.log_hp)

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
        lowest_train, lowest_val, batch_size_train, batch_size_val, batch_num_train, batch_num_val = train(config=config, tunable_config=t_config, device=device,
                                                                                                           train_loader=train_loader, val_loader=val_loader, scaler=scaler,
                                                                                                           train_writer=train_writer, val_writer=val_writer, graph_writer=graph_writer, hp_writer=hp_writer,
                                                                                                           batch_train_writer=batch_train_writer, batch_val_writer=batch_val_writer,
                                                                                                           model_save=meta_save.save_model, tensors_save=meta_save.save_tensors,
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
        test_dict, batch_size_test, batch_num_test, output_ls, output_raw_ls = test(config=config, test_loader=test_loader, scaler=scaler, device=device)
        testing_time = time.time() - testing_time_0
        # print(testing_time) # seconds
        print(str(datetime.timedelta(seconds=testing_time))) # days, hours:minutes:seconds

        meta_save.output_metadata_test(test_dict=test_dict, computation_time=testing_time, 
                                    batch_size=batch_size_test, batch_num=batch_num_test)
        
        meta_save.save_tensors(output_ls, 'output_ls.pt')
        meta_save.save_tensors(output_raw_ls, 'output_raw_ls.pt')

        print(f'test metrics: {test_dict}')

def loop_over_supp_vars(config, counter, supp_vars, skip_vars_keys=[4, 5, 6]):
    for key, supp_var in supp_vars.items():
        config.process = counter
        counter += 1

        # skip certain supp_vars
        if key in skip_vars_keys:
            continue

        config.dataset_path = []
        for var in supp_var:
            path = f'{data_root_dir}/{region}_{var}'
            config.dataset_path.append(path)
        
        config.input_dim = len(config.dataset_path)
        config.target_dim = len(config.dataset_path) 
        print(f'target len: {config.trg_seq_len}, num_test_samples {config.num_test_samples}, input_dim {config.input_dim}, process {config.process}')

        # train
        config.train_mode = True
        config.load_net_path = None
        main(config, t_config)
        print('=============================Training finished=============================')

        # test
        config.train_mode = False
        config.load_net_path = os.path.join(config.root_dir, config.model_name+f'{config.process:03d}', 'model/best_loss_model_.pth')
        print(config.load_net_path)
        main(config, t_config)
        print('=============================Testing finished=============================') 




if __name__ == '__main__':

    config, t_config = configs([0])

    data_root_dir = '../data/data_processing_v2'
    supp_vars_eu = {0: ['incidence_rate_no_index.csv'],
                    1: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv'],
                    2: ['incidence_rate_no_index.csv', 'hospitalization_rate_no_index.csv'],
                    3: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv'],
                    4: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv',
                        'vaccination_rate_no_index.csv'], 
                    5: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv',
                        'vaccination_rate_no_index.csv', 'testing_no_index.csv'],
                    6: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv',
                        'vaccination_rate_no_index.csv', 'testing_no_index.csv', 'airtraffic_increase_rate_no_index.csv']}
    supp_vars_us = {0: ['incidence_rate_no_index.csv'],
                    1: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv'],
                    2: ['incidence_rate_no_index.csv', 'hospitalization_rate_no_index.csv'],
                    3: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv'],
                    4: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv',
                        'vaccination_rate_no_index.csv'], 
                    5: ['incidence_rate_no_index.csv', 'fatality_rate_no_index.csv', 'hospitalization_rate_no_index.csv',
                        'vaccination_rate_no_index.csv', 'testing_no_index.csv']}

    if config.sensors == 28:
        region = 'eu'
        # loop over folds + test
        for fold_index in range(config.num_folds): 
            if fold_index == config.num_folds-1: # test mode
                config.split_by_fold = False
                config.num_test_samples = None
                config.root_dir = f'../logs/linears/{region}/test'
            else:
                config.split_by_fold = True
                config.current_fold = fold_index
                config.root_dir = f'../logs/linears/{region}/fold{fold_index}'
            
            # loop over target lengths: non-seasonal
            counter = 1
            for config.trg_seq_len in [3, 6, 12]: 
                config.src_seq_len = 12
                config.past_history_factor = config.src_seq_len//config.trg_seq_len
                config.max_len = config.trg_seq_len

                if config.split_by_fold: # fold mode
                    if config.trg_seq_len == 12:
                        config.num_test_samples = 101
                    else:
                        config.num_test_samples = 102 
                else:
                    config.num_test_samples = None # test mode
                
                loop_over_supp_vars(config, counter, supp_vars_eu, skip_vars_keys=[4, 5, 6])
                counter += len(supp_vars_eu)
            
            # loop over target lengths: seasonal
            for config.trg_seq_len in [2, 7, 14]: 
                config.src_seq_len = 14
                config.past_history_factor = config.src_seq_len//config.trg_seq_len
                config.max_len = config.trg_seq_len

                if config.split_by_fold: # fold mode
                    if config.trg_seq_len == 14:
                        config.num_test_samples = 101
                    else:
                        config.num_test_samples = 102 
                else:
                    config.num_test_samples = None # test mode
                
                loop_over_supp_vars(config, counter, supp_vars_eu, skip_vars_keys=[4, 5, 6])
                counter += len(supp_vars_eu)

    elif config.sensors == 49:
        region = 'us'
        for fold_index in range(config.num_folds):
            if fold_index == config.num_folds-1: # test mode
                config.split_by_fold = False
                config.num_test_samples = None
                config.root_dir = f'../logs/linears/{region}/test'
            else:
                config.split_by_fold = True
                config.current_fold = fold_index
                config.root_dir = f'../logs/linears/{region}/fold{fold_index}'

            counter = 1
            for config.trg_seq_len in [3, 6, 12]: # non-seasonal
                config.src_seq_len = 12
                config.past_history_factor = config.src_seq_len//config.trg_seq_len
                config.max_len = config.trg_seq_len

                if config.split_by_fold: # fold mode
                    if config.trg_seq_len == 12:
                        config.num_test_samples = 101
                    else:
                        config.num_test_samples = 102 
                else:
                    config.num_test_samples = None # test mode
                
                loop_over_supp_vars(config, counter, supp_vars_us, skip_vars_keys=[4, 5])
                counter += len(supp_vars_us)

            for config.trg_seq_len in [2, 7, 14]: # seasonal
                config.src_seq_len = 14
                config.past_history_factor = config.src_seq_len//config.trg_seq_len
                config.max_len = config.trg_seq_len

                if config.split_by_fold: # fold mode
                    if config.trg_seq_len == 14:
                        config.num_test_samples = 101
                    else:
                        config.num_test_samples = 102 
                else:
                    config.num_test_samples = None # test mode
                
                loop_over_supp_vars(config, counter, supp_vars_us, skip_vars_keys=[4, 5])
                counter += len(supp_vars_us)
    else:
        raise ValueError('double check for sensor parameter in config!')
