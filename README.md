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
```bash
pip install -r requirements.txt
```

## Where to find data
You can obtained the well pre-processed datasets from [Google Drive] or [Baidu Drive]. Then place the downloaded data under the folder ./data

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