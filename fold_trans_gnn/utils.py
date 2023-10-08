import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
import torch
import os
import random

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype, num_nodes=5):
    if adjtype == "init_from_scratch":
      adj = [np.diag(np.ones(num_nodes)).astype(np.float32)]
      return adj
    else:
      sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
      if adjtype == "scalap":
          adj = [calculate_scaled_laplacian(adj_mx)]
      elif adjtype == "normlap":
          adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
      elif adjtype == "symnadj":
          adj = [sym_adj(adj_mx)]
      elif adjtype == "transition":
          adj = [asym_adj(adj_mx)]
      elif adjtype == "doubletransition":
          adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
      elif adjtype == "identity":
          adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
      else:
          error = 0
          assert error, "adj type not defined"
      return sensor_ids, sensor_id_to_ind, adj

# threshold and sparsity ratio
def threshold_sparsity(supports):
    num_sensors = supports.shape[0]
    threshold = round(1/num_sensors, 4)
    nonzero = torch.count_nonzero(supports)
    sparsity = round(nonzero.item() / (num_sensors * num_sensors), 4)
    return threshold, sparsity

# pkl_filename = '/nas/guosuhan/gwnet/dataset/covid-eu/version2/eu_adj_covid.pkl'
# pkl_filename = '/root/autodl-tmp/dataset/covid-eu/adj_covid_google_map.pkl' 
# pkl_filename = '/nas/guosuhan/gwnet/dataset/covid-eu/version2/us_adj_covid.pkl'
# adj_mx = load_pickle(pkl_filename)
# supports = torch.from_numpy(adj_mx)
# thres, sp = threshold_sparsity(supports)
# print(f'current adj has threshold {thres} and sparsity {sp}')
# current adj has threshold 0.0357 and sparsity 0.4069