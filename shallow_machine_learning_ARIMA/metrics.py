import numpy as np
import torch

""" metrics """
# def mse(preds, labels, mask=False, null_val=np.nan):
#     loss = (preds - labels) ** 2 
    
#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
    
#     return torch.mean(loss)

def mse(preds, labels, mask=False, null_val=np.nan):
    loss = (preds - labels) ** 2 
    
    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds-labels)**2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
    return torch.mean(loss)
    

def rmse(preds, labels, mask=False, null_val=np.nan):
    return torch.sqrt(mse(preds, labels, mask, null_val))


# def mae(preds, labels, mask=False, null_val=np.nan):
#     loss = torch.abs(preds - labels)

#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
    
#     return torch.mean(loss)


def mae(preds, labels, mask=False, null_val=np.nan):
    loss = torch.abs(preds - labels)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


# def mape(preds, labels, mask=False, null_val=np.nan):
#     loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)

#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)

#     return torch.mean(loss)


def mape(preds, labels, mask=False, null_val=np.nan):
    loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)/labels
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def mape_clipped(preds, labels):
    loss = torch.abs(preds - labels)/(torch.abs(labels) + 1e-5)
    loss = torch.where(loss.double() > 5., 5., loss.double()).float()
    
    return torch.mean(loss)


# def smape(preds, labels, mask=False, null_val=np.nan):
#     loss = 2.0 * (torch.abs(preds - labels)) / (torch.abs(preds) + torch.abs(labels) + 1e-5)

#     if mask:
#         if np.isnan(null_val):
#             loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  
#         else:
#             loss = torch.where((labels==null_val), torch.zeros_like(loss), loss)
    
#     return torch.mean(loss) 

def smape(preds, labels, mask=False, null_val=np.nan):
    loss = 2.0 * (torch.abs(preds - labels)) / (torch.abs(preds) + torch.abs(labels) + 1e-5)

    if mask:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
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

def build_metric(y_pred, y_true, mask=False, null_val=np.nan, 
                metric_key_list=['mae', 'mae_mask', 'mse', 'mse_mask', 'rmse', 'rmse_mask', 'mape', 'mape_clip', 'mape_mask','smape']):

    assert type(metric_key_list) == list, 'metric_key_list has wrong input type, must be list of strings'

    metrics_reported = {}
    for metric_key in metric_key_list:
        if metric_key == 'mape_clip':
           metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true).item() 
        elif 'mask' in metric_key:
            table_key, _ = metric_key.split('_')
            metrics_reported[metric_key] = metrics_table[table_key](y_pred, y_true, True, 0.0).item() 
        else:
            # assign float from tensor
            metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true, mask, null_val).item()

    return metrics_reported


# def build_metric(y_pred, y_true, mask=None, null_val=np.nan, 
#                 metric_key_list=['mae', 'mse', 'rmse', 'mape', 'mape_clip', 'mape_mask','smape']):

#     assert type(metric_key_list) == list, 'metric_key_list has wrong input type, must be list of strings'

#     metrics_reported = {}
#     for metric_key in metric_key_list:
#         if metric_key == 'mape_clip':
#            metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true).item() 
#         elif metric_key == 'mape_mask':
#             metrics_reported[metric_key] = metrics_table['mape'](y_pred, y_true, True, 0.0).item()
#         else:
#             # assign float from tensor
#             metrics_reported[metric_key] = metrics_table[metric_key](y_pred, y_true, mask, null_val).item()

#     return metrics_reported

# preds = torch.rand(4, 10, 3, 1)
# labels = torch.ones(4, 10, 3, 1)
# na_labels = torch.ones(4, 10, 3, 1)
# na_labels[:, :, 1, :] = np.nan
# labels_0 = torch.ones(4, 10, 3, 1)
# labels_0[:, :, 1, :] = 0

# out = smape(preds, labels)
# out1 = smape(preds, na_labels, True)
# out2 = smape(preds, labels_0, True, 0)
# print(f'w/o mask {out:.4f}, with nan mask {out1:.4f}, with 0 mask {out2:.4f}')

# metrics_repo = build_metric(preds, labels_0)
# metrics_repo_1 = build_metric(preds, na_labels, True)
# metrics_repo_2 = build_metric(preds, labels_0, True, 0)

# temp_dict = {key : round(metrics_repo[key], 4) for key in metrics_repo}