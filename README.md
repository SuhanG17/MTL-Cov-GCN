# MTL-Cov-GCN
A repository for the code for Cov-GCN, MTL-Cov_GCN and related experiments

## Folder Introduction
```bash
├── data
│   ├── data_processing_v2/ # where data should be stored
│   └── data_processing.ipynb # how data is preprocessed
├── fold_linear_layer
│   ├── combine_csv.py # aggregate performances for many models into a single csv file, path_dir has to be passed 
│   ├── config.py # where parameters are stored, annotations on parameters are given as comment
│   └── dataset.py # data is repackaged into dataloader object
│   └── json_to_csv.py # retrieve performances from where metrics are saved, path_dir has to be passed 
│   └── linear_layers.py # network
│   └── logger.py # how to save model, tensorboard writer, output tensor and more
│   └── loss_and_optimizer.py # construct loss function and optimizer
│   └── metrics.py # calculate metrics
│   └── MLM.py # masking part of the input, function NOT used in this research
│   └── trainer_single_process.py # main script to run if train for a sinlge model
│   └── trainer.py # main script to run if train for a multiple models in a loop
│   └── utils.py # auxilliary tools, set random seed, calculate sparsity and threshold, etc.
├── dir3
├── file_in_root.ext
└── README.md
```

## How to set up environment?

## Where to find data

## How to run experiments?

## Where is model and metrics saved?

## What do parameters in configs stands for?

## If I want to use my own data, what parameters should I tune?
