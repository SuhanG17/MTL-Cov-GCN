import numpy as np
import torch

""" metrics """

def mse(preds, labels, mask=False, null_val=np.nan, raw_labels=None):
    loss = (preds - labels) ** 2 
    
    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            if (raw_labels is not None):
                mask = (raw_labels!=null_val)
            else:
                mask = (labels!=null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds-labels)**2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
    return torch.mean(loss)
    

def rmse(preds, labels, mask=False, null_val=np.nan, raw_labels=None):
    return torch.sqrt(mse(preds, labels, mask, null_val, raw_labels))


def mae(preds, labels, mask=False, null_val=np.nan, raw_labels=None):
    loss = torch.abs(preds - labels)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            if (raw_labels is not None):
                mask = (raw_labels!=null_val)
            else:
                mask = (labels!=null_val)
        mask = mask.float()
        s = torch.mean((mask))
        # print(s)
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def mae_m(preds, labels, mask=True, null_val=np.nan, raw_labels=None):
    loss = torch.abs(preds - labels)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            if (raw_labels is not None):
                mask = (raw_labels!=null_val)
            else:
                mask = (labels!=null_val)
        mask = mask.float()
        s = torch.mean((mask))
        # print(s)
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def mape(preds, labels, mask=False, null_val=np.nan, raw_labels=None):
    loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)
    # sum_ = torch.sum(torch.where(np.isinf(loss), torch.ones_like(labels), torch.zeros_like(labels)))
    # print(sum_)
    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            if (raw_labels is not None):
                mask = (raw_labels!=null_val)
            else:
                mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)/(torch.abs(labels) + 1e-5)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # print(torch.mean(loss))
    return torch.mean(loss)


def mape_clipped(preds, labels):
    loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)
    loss = torch.where(loss.double() > 5., 5., loss.double()).float()
    
    return torch.mean(loss)


def smape(preds, labels, mask=False, null_val=np.nan, raw_labels=None):
    loss = 2.0 * (torch.abs(preds - labels)) / (torch.abs(preds) + torch.abs(labels) + 1e-5)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            if (raw_labels is not None):
                mask = (raw_labels!=null_val)
            else:
                mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = 2.0 * (torch.abs(preds - labels)) / (torch.abs(preds) + torch.abs(labels) + 1e-5)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
 
    return torch.mean(loss) 



""" metrics_table"""

metrics_table = {'mae': mae,
                 'mse': mse,
                 'rmse': rmse,
                 'mape': mape,
                 'mape_clip': mape_clipped,
                 'smape': smape}

""" build metrics fn"""

def build_metric(y_pred, y_true, mask=False, null_val=np.nan, raw_labels=None,
                metric_key_list=['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape']):

    assert type(metric_key_list) == list, 'metric_key_list has wrong input type, must be list of strings'

    metrics_reported = {}
    for metric_key in metric_key_list:
        if metric_key == 'mape_clip':
           metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true).item() 
        elif 'mask' in metric_key:
            table_key, _ = metric_key.split('_')
            metrics_reported[metric_key] = metrics_table[table_key](y_pred, y_true, True, 0.0, raw_labels).item() 
        else:
            # assign float from tensor
            metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true, mask, null_val, raw_labels).item()

    return metrics_reported

# def mse(preds, labels, mask=False, null_val=np.nan):
#     loss = (preds - labels) ** 2 
    
#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
    
#     return torch.mean(loss)
    

# def rmse(preds, labels, mask=False, null_val=np.nan):
#     return torch.sqrt(mse(preds, labels, mask, null_val))


# def mae(preds, labels, mask=False, null_val=np.nan):
#     # print(preds, labels)
#     loss = torch.abs(preds - labels)

#     if mask:
#         # print(loss)
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
        
#     return torch.mean(loss)

# def mape(preds, labels, mask=False, null_val=np.nan):
#     loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)

#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)

#     return torch.mean(loss)

# def mape_clipped(preds, labels):
#     loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)
#     loss = torch.where(loss.double() > 5., 5., loss.double()).float()
    
#     return torch.mean(loss)


# def smape(preds, labels, mask=False, null_val=np.nan):
#     loss = 2.0 * (torch.abs(preds - labels)) / (torch.abs(preds) + torch.abs(labels) + 1e-5)

#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
    
#     return torch.mean(loss) 


# """ metrics_table"""

# metrics_table = {'mae': mae,
#                  'mse': mse,
#                  'rmse': rmse,
#                  'mape': mape,
#                  'mape_clip': mape_clipped,
#                  'smape': smape}

# """ build metrics fn"""

# def build_metric(y_pred, y_true, mask=False, null_val=np.nan, 
#                 metric_key_list=['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape']):

#     assert type(metric_key_list) == list, 'metric_key_list has wrong input type, must be list of strings'

#     metrics_reported = {}
#     for metric_key in metric_key_list:
#         if metric_key == 'mape_clip':
#            metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true).item() 
#         elif 'mask' in metric_key:
#             table_key, _ = metric_key.split('_')
#             metrics_reported[metric_key] = metrics_table[table_key](y_pred, y_true, True, 0.0).item() 
#         else:
#             # assign float from tensor
#             metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true, mask, null_val).item()

#     return metrics_reported

# # [batch_size, seq_len, dim] ->  [batch_size, dim, seq_len]
# batch_size = 4
# seq_len = 3
# dim = 1
# torch.manual_seed(0)
# preds = torch.rand(batch_size, seq_len, dim)
# # labels = torch.ones(batch_size, seq_len, dim)
# # print(mse(preds, labels, True, 0))
# # print(mse(preds.permute(0, 2, 1), labels.permute(0, 2, 1), True, 0))

# # na_labels = torch.ones(batch_size, seq_len, dim)
# # na_labels[:, 1, :] = np.nan
# labels_0 = torch.ones(batch_size, seq_len, dim)
# labels_0[:, 1, :] = 0

# # print(mape(preds, na_labels, True, 0.0))
# print(mape(preds, labels_0, True, 0.0))
# print(mape(preds.permute(0, 2, 1), labels_0.permute(0, 2, 1), True, 0.0))