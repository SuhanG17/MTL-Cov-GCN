import collections
import copy
import torch
import time
import datetime
from tqdm import tqdm
import torch.nn.functional as F

from utils import set_seed
from dataset import build_dataset, Batch
from original_transformer import build_network, count_parameters, batch_greedy_decode
from metrics import build_metric
from loss_and_optimizer import build_optimizer_linear, build_loss, EarlyStopping
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


def train_epoch(network, loader, loss_fn, scaler, opt_and_scheduler, optimizer_method, norm_method, device, metric_key_list, batch_writer=None):
    cumu_loss = 0
    cumu_metric = {}

    for sample_id, (inputs, targets, targets_raw) in enumerate(loader):
        inputs = inputs.to(device) # [batch_size, seq_len, dim] 
        targets = targets.to(device) # [batch_size, seq_len, dim] 
        # print(inputs.shape, targets.shape)
        # print(f'inputs has shape {inputs.shape}, targets has shape {targets.shape}')

        # create mask using Batch class
        batch = Batch(inputs, targets, pad=-10, device=device)

        # zero the parameter gradients
        if optimizer_method=='WarmUp':
            opt_and_scheduler.optimizer.zero_grad()
        else:
            opt_and_scheduler.zero_grad()

        # ➡ Forward pass
        outputs = network(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # print(outputs)
        # loss = loss_fn(outputs, batch.trg_y)
        loss = loss_fn(outputs, batch.trg_y, mask=True, null_val=0.0, raw_labels=targets_raw.to(device))
        # print("loss without weights: {}".format(loss))

        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        opt_and_scheduler.step()

        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)

        # metrics
        with torch.no_grad():
            outputs_raw = scaler.inverse_transform(outputs.to('cpu'), norm_method)
            metrics = build_metric(outputs_raw, targets_raw, metric_key_list=metric_key_list)

        if not cumu_metric: # if cumu_metric is an empty dict
            cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
        else:
            cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]

        if batch_writer:
            loss_dict = {"batch_train_loss": loss.item()}
            metrics.update(loss_dict)
            log_writer_batch(batch_writer, metrics, sample_id)

    return cumu_loss / len(loader), dict_values_division(cumu_metric, len(loader))

def dev_epoch(network, loader, loss_fn, scaler, norm_method, device, metric_key_list, batch_writer=None):
    cumu_loss = 0
    cumu_metric = {}

    with torch.no_grad():
        for i, (inputs, targets, targets_raw) in enumerate(loader):
            # [batch_size, seq_len, dim] 
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # create mask using Batch class
            batch = Batch(inputs, targets, pad=-10, device=device)

            # ➡ Forward pass only
            outputs = network(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_fn(outputs, batch.trg_y, mask=True, null_val=0.0, raw_labels=targets_raw.to(device))
            # loss = loss_fn(outputs, batch.trg_y)
            # print("loss without weights: {}".format(loss))
            cumu_loss += loss.item()

            # metrics
            outputs_raw = scaler.inverse_transform(outputs.to('cpu'), norm_method)
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

def train(config=None, tunable_config=None, device='cpu', 
          train_loader=None, val_loader=None, scaler=None,
          train_writer=None, val_writer=None, graph_writer=None, hp_writer=None,
          batch_train_writer=None, batch_val_writer=None,
          model_save=None, early_stopping=None):

    # train_loader, val_loader, _, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
    #                                          config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last)    
    network = build_network(config.input_dim, config.target_dim, 
                            tunable_config.N, tunable_config.d_model, tunable_config.d_ff, tunable_config.h, tunable_config.dropout, 
                            device, True, config.load_net_path)
    print(f'current network has {count_parameters(network)} parameters')
    opt_and_scheduler = build_optimizer_linear(network, config.lr, t_config.optimizer_method, config.factor, config.warmup, config.resume_from_step)
    loss_fn = build_loss(config.loss_type, False)

    # initiate log to document the lowest loss/metrics
    lowest_train = initialize_lowest_log_dict(config.metric_key_list)
    lowest_val = initialize_lowest_log_dict(config.metric_key_list)

    for epoch in range(config.epochs):
        network.train()
        avg_loss_train, avg_metrics_train = train_epoch(network, train_loader, loss_fn, scaler, opt_and_scheduler, t_config.optimizer_method, config.norm_method, device, config.metric_key_list, batch_train_writer)
        loss_dict_train = {"train_loss": avg_loss_train}
        avg_metrics_train.update(loss_dict_train)
        if train_writer:
            log_writer_epoch(train_writer, avg_metrics_train, epoch)

        network.eval()    
        avg_loss_val, avg_metrics_val = dev_epoch(network, val_loader, loss_fn, scaler, config.norm_method, device, config.metric_key_list, batch_val_writer)
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
            model_save(network, num_epoch=epoch, total_epochs=config.epochs, Parallel=True if torch.cuda.device_count() > 1 else False)
            # True if torch.cuda.device_count() > 1 else False
        
                # early stopping
        if early_stopping:
            early_stopping(avg_metrics_train['train_loss'], avg_metrics_val['val_loss'])
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
    
    
    if graph_writer:
        dummy_inputs = torch.ones(config.batch_size[0], config.src_seq_len, config.input_dim).to(device)
        dummy_targets = torch.ones(config.batch_size[0], config.trg_seq_len, config.target_dim).to(device)
        batch = Batch(dummy_inputs, dummy_targets, pad=-10, device=device)

        if torch.cuda.device_count() > 1:
            graph_writer.add_graph(network.module, [batch.src, batch.trg, batch.src_mask, batch.trg_mask])
        else:
            graph_writer.add_graph(network, [batch.src, batch.trg, batch.src_mask, batch.trg_mask])
        graph_writer.close()
    
    if hp_writer:
        log_writer_hparams(hp_writer, tunable_config, avg_metrics_val)
    
    return lowest_train, lowest_val, train_loader.batch_size, val_loader.batch_size, len(train_loader), len(val_loader)
    

def test(config=None, tunable_config=None, test_loader=None, scaler=None, device='cpu'):

    # _, _, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
    #                                          config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last)       
    network = build_network(config.input_dim, config.target_dim, 
                            tunable_config.N, tunable_config.d_model, tunable_config.d_ff, tunable_config.h, tunable_config.dropout, 
                            device, False, config.load_net_path)
    network.eval()
    print(f'current network has {count_parameters(network)} parameters')
    loss_fn = build_loss(config.loss_type, False)

    cumu_loss = 0
    cumu_metric = {}

    out_list = []

    with torch.no_grad():
        for _, (inputs, targets, targets_raw) in enumerate(test_loader):
            # [batch_size, seq_len, dim] 
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # create mask using Batch class
            batch = Batch(inputs, targets, pad=-10, device=device)
 
            outputs =  batch_greedy_decode(network, batch.src, batch.src_mask, config.trg_seq_len, batch.start_token, device)
            loss = loss_fn(outputs, batch.trg_y, mask=True, null_val=0.0, raw_labels=targets_raw.to(device))
            # loss = loss_fn(outputs, batch.trg_y)

            cumu_loss += loss.item() 

            # metrics
            outputs_raw = scaler.inverse_transform(outputs.to('cpu'), config.norm_method)
            out_list.append(outputs_raw)
            metrics = build_metric(outputs_raw, targets_raw, metric_key_list=config.metric_key_list)

            if not cumu_metric: # if cumu_metric is an empty dict
                cumu_metric = copy.deepcopy(metrics) # deepcopy() creates new object rather than pointer
            else:
                cumu_metric = list(cumulative_elementwise_sum([cumu_metric, metrics]))[-1]

    loss_dict = {'test_loss': cumu_loss / len(test_loader)}
    test_dict = dict_values_division(cumu_metric, len(test_loader)) 
    test_dict.update(loss_dict)

    return out_list, test_dict, test_loader.batch_size, len(test_loader)

def generate_loader(config=None, tensors_save=None, fold=0):
    if config.split_by_fold:
        train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                     config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
                                                                     fold=fold, num_folds=config.num_folds, num_test_samples=config.num_test_samples, save_tensor=tensors_save)    
    else:
        train_loader, val_loader, test_loader, scaler = build_dataset(config.dataset_path, config.norm_method, config.past_history_factor, config.trg_seq_len, config.target_dim,
                                                                      config.rand, config.ratio, config.batch_size, config.shuffle, config.drop_last, 
                                                                      fold=None, num_folds=config.num_folds, num_test_samples=config.num_test_samples, save_tensor=tensors_save,)

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # set seed
    set_seed(17)

    # init configs
    config, t_config = configs([1, 2, 2, 0, 0, 2])
    torch.set_printoptions(precision=6)

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


        # train
        training_time_0 = time.time()
        train_loader, val_loader, test_loader, scaler = generate_loader(config, meta_save.save_tensors, fold=config.current_fold)
        lowest_train, lowest_val, batch_size_train, batch_size_val, batch_num_train, batch_num_val = train(config=config, tunable_config=t_config, device=device, 
                                                                                                           train_loader=train_loader, val_loader=val_loader, scaler=scaler,
                                                                                                           train_writer=train_writer, val_writer=val_writer, graph_writer=graph_writer, hp_writer=hp_writer,
                                                                                                           batch_train_writer=batch_train_writer, batch_val_writer=batch_val_writer,
                                                                                                           model_save=meta_save.save_model, early_stopping=early_stopping)
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
        out_list, test_dict, batch_size_test, batch_num_test = test(config=config, tunable_config=t_config, test_loader=test_loader, scaler=scaler, device=device)
        testing_time = time.time() - testing_time_0
        # print(testing_time) # seconds
        print(str(datetime.timedelta(seconds=testing_time))) # days, hours:minutes:seconds

        meta_save.output_metadata_test(test_dict=test_dict, computation_time=testing_time, 
                                       batch_size=batch_size_test, batch_num=batch_num_test)
        meta_save.save_tensors(out_list, 'outputs_raw.pt')