# MTL-Cov-GCN
A repository for the code for Cov-GCN, MTL-Cov_GCN and related experiments

## Folder Introduction
```bash
├── [data] # to be initiated
│   ├── data_processing_v2/ # where data should be stored
│   └── data_processing.ipynb # how data is preprocessed
├── [logs] # to be initated
├── fold_linear_layer
│   ├── config.py # where parameters are stored, annotations on parameters are given as comment
│   └── dataset.py # data is repackaged into dataloader object
│   └── linear_layers.py # network
│   └── logger.py # how to save model, tensorboard writer, output tensor and more
│   └── loss_and_optimizer.py # construct loss function and optimizer
│   └── metrics.py # calculate metrics
│   └── MLM.py # masking part of the input, function NOT used in this research
│   └── trainer_single_process.py # main script to run if train for a sinlge model
│   └── trainer.py # main script to run if train for a multiple models in a loop
│   └── utils.py # auxilliary tools, set random seed, calculate sparsity and threshold, etc.
│   └── run.sh # bash script to train or test model, instructions inside
├── fold_trans
├── fold_trans_gnn
│   ├── beta_controller.py # changing beta, function NOT used in this research
│   └── ... 
├── fold_trans_gnn_adp_map
├── shallow_machine_learning_ARIMA
│   ├── param_search_feature_remove.py # search the best parameters for XGBoost, LightGBM, RandomForest and ARIMA
│   └── ... 
└── README.md
```

## How to set up environment?
Install Python 3.8.
0. Preferably with conda environment, run
```bash
conda create -n gnn python=3.8 # create virtual environment with name gnn
conda activate gnn # go into virtual env. CAUTION: all installation below should happen inside the env
conda deactivate # exit virtual env
```
[OPTIONAL] For users from Mainland, China, add channel to speed-up package installation
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
1. Install 1.10.0 torch GPU version with CUDA=11.1, run
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Install torch_scatter, torch_sparse, they are prerequiste for torch_geometric. 
    - Because they have strict version control w.r.t to torch version and cuda version, we provide path to the wheel file for installation. [torch_scatter](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl) [torch_sparse](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl)
    - place .whl files inside the .whls/ 
    - For other CUDA version please refer to [Documentaion](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) under **Installation from Wheels -> Note -> You can look up the latest supported version number here.**
```bash
pip install ../whls/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install ../whls/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
```

3. Install torch_geometric for torch==1.10.0 with CUDA=11.1
```bash
pip install torch_geometric==2.0.2
```

4. Install other pacakges
```bash
pip install -r requirements.txt
```

## Where to find data
You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1nrbJmSl1X7IbnPliXeug7b7iCwKy8pqm?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1e9Blw7nsvPbUyM-ecFzPEQ) with four-character password: hmlr . Then place the downloaded data under the folder ./data

## How to run experiments?
Run the bash file in each folder to run experiments.

## Where is model and metrics saved?
They will be save in the logs/ folder with path assigned in the config.py 

## What do parameters in configs stands for?
Annotation to parameters are in the config.py.

## If I want to use my own data, what parameters should I tune?
```python
# change to your own
'dataset_path'
'pkl_filename'
'sensors'
'input_dim'
'target_dim'
'src_seq_len'
'trg_seq_len'
'past_history_factor'
'max_len'
# need to fine-tune by hand
'threshold'
'sparsity_ratio'
'factor'
'warmup'
```