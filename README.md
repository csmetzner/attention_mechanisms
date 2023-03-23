# Attention Mechanisms in Clinical Text Classification: A Comparative Analysis
This is the GitHub repository associated with the paper. 

## Project Introduction, Objectives, Methods, and Result
Attention mechanisms became a mainstay architecture in present neural networks, improving the performance in biomedical text classification tasks. In particular, models performing automated medical encoding of clinical documents make extensive use of the label-wise attention mechanism. Label-wise attention mechanism increases a model's discriminatory ability by utilizing label-specific reference information. This information can either be implicitly learned during training or explicitly provided through textual code descriptions or code hierarchy embeddings; however, contemporary work selects the type of label-specific reference information arbitrarily. Therefore, in this work, we evaluated label-wise attention initialized either with implicit random or explicit pretrained label-specific reference information against two common baseline methods, target attention and text-encoder architecture-specific methods to generate document embeddings, across four text-encoder architectures, which are a convolutional neural network, two recurrent neural networks, and a Transformer. We additionally introduce a hierarchical extension of label-wise attention designed to incorporate explicit information on code hierarchy. We performed our experiments on MIMIC-III, a standard dataset in the clinical text classification domain. Our experimental results showed that using pretrained reference information and the hierarchical design helped improve classification performance but had a diminishing effect with an increasing number of samples and larger label spaces across all text-encoder architectures. 

All models and experiments were implemented in Python 3.8 and PyTorch 1.12. 

## Dependencies
* python 3.8
* numpy 1.23.0
* pandas 1.4.2
* scikit-learn 1.1.1
* gensim 4.2.0
* nltk 3.7
* tqdm 4.64.0
* xlsxwriter 3.0.3
* openpyxl 3.0.10
* transformers 4.19.2
* torch 1.13.1
* torchtext 0.14.0
* hp5y 3.7.0

** Note: ** These are the versions last tested, but earlier version may work as well.

## Repository Structure
```
├── data
│   ├── raw                                       <- The original data downloaded from https://physionet.org/content/mimiciii/1.4/
│   ├── processed                                 <- The processed data used to train and evaluate the models.
│   │   └── hadm_ids                              <- Hospital admission ids defining the training, testing, and validation splits for MIMIC-III-Full and MIMIC-III-50 retrieved from https://github.com/jamesmullenbach/caml-mimic/tree/master/mimicdata/mimic3.
│   └── external                                  <- External data describing ICD-9 codes and their descriptions.
│       ├── CMS32_DESC_LONG_SHORT_DX.xlsx
│       ├── CMS32_DESC_LONG_SHORT_SG.xlsx
│       ├── D_ICD_DIAGNOSES.csv
│       └── D_ICD_PORCEDURES.csv
│
├── notebooks
│   ├── notebook_discussion.ipynb                 <- Contains code and results of analysis in discussion sections B and D.
│   ├── notebook_preprocessing_MIMIC_III_and_code_descriptions.ipynb      <- Contains code to preprocess the raw MIMIC-III clinical documents and ICD-9 code descriptions.
│   └── notebook_results_Mimic_III.ipynb          <- Contains code used to visualize results.
│
├── src
│   ├── attention_modules                         <- Source code for attention mechanisms.
│       ├── attention_mechanisms.py               <- Script that functions as a "middle-man" between the actual attention mechanisms and the source code for the models in the directory "models".
│       ├── hierarchical_attention.py             <- Contains source code for the hierarchical pretrained / random label-attention mechanism.
│       ├── multihead_attention.py                <- Contains source code for the multi-head attention mechanism retrieved from https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html.
│       ├── pretrained_attention.py               <- Contains source code for the pretrained label-attention mechanism.
│       ├── random_attention.py                   <- Contains source code for the random label-attention mechanism.
│       ├── target_attention.py                   <- Contains source code for target-attention mechanism.
│       └── __init__.py
│   ├── config_files                              <- Contains yaml-based config files for each model; each file contains model_kwargs .and train_kwargs; model_kwargs determine the design of the model architecture (e.g., size of hidden dimension) and train_kwargs determine the hyperparameters for the training (e.g., learning rate or batch size).
│       ├── BiGRU_config.yml                      
│       ├── BiLSTM_config.yml
│       ├── CLF_config.yml
│       ├── CNN_config.yml
│       └── datasets_config.yml                   <- Contains information about the label spacD-9 categories) and maximal document length. 
│   ├── models                                    <- Contains source code for the models / text-encoder architectures used in the study.
│       ├── CNN.py
│       ├── RNN.py                                <- Contains source code for both the BiLSTM and BiGRU text-encoder architectures.
│       ├── Transformers.py                       <- Contains source code for the HuggingFace Transformer **"Clinical Longformer"**; If you want to run this model you need to first download the model from https://huggingface.co/yikuan8/Clinical-Longformer, then change the path in line 87 to point to the location of the donwloaded directory containing the source code, vocabulary, and word embeddings for the Clinical Longformer.
│       └── __init__.py
│   ├── tools                                     <- Contains source code to run the experiments (training.py, dataloaders.py, performance_metrics.py, utils.py), for the analysis (analysis_energyscores.py, print_results.py), and for preprocessing the raw data (utils_preprocess.py).
│       ├── analysis_energyscores.py              <- Contains analysis to retrieve phrase-level extraction and post-process energy scores.
│       ├── dataloaders.py                        <- Contains source code for pytorch dataloader object.
│       ├── performance_metrics.py                <- Contains source code for all performance metrics used in the study (F1-scores, precision@k, AUC).
│       ├── print_results.py                      <- Contains source code that helps printing results on terminal.
│       ├── training.py                           <- Contains source code controlling the training procedure.
│       ├── utils.py                              <- Contains source code for generating word embedding matrices and help argparse command line arguments.
│       ├── utils_preprocess.py                   <- Contains source code to preprocess the clinical notes - scripts are called by the notebook * notebook_preprocessing_MIMIC_III_and_code_descriptions.ipynb *
│       └── __init__.py
│   ├── .gitkeep
│   ├── __init__.py
│   └── main.py                                   <- Contains source code to run the model experiments   
│
├── .gitignore
├── Pipfile
├── Pipfile.lock
└── README.md
```

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. The raw data were retrieved from https://physionet.org/content/mimiciii/1.4/ - the licence needs to be requested

## Replicating the Results and Run the Experiments
### Preprocessing of Raw Data
1. Move the raw data files with the names *NOTEEVENTS.csv*, *DIAGNOSES_ICD.csv.gz*, and *PROCEDURES_ICD.csv.gz to the /root/data/raw/
2. Run all cells of the notebook **notebook_preprocessing_MIMIC_III_and_code_descriptions.ipynb**

### Running a model ###
For example, to run a model using a CNN and random label-attention on the MIMIC-III-50 subset you would do the following:

`python -d Mimic50 -m CNN -am random -en first_model -cq True -ci True -as True` 

The line above would create a directory named "results_first_model" containing sub-directories containing models, predictions, and scores. Models contains the models ran by you, predictions contains the sigmoid probabilities, and scores will contain the performance scores overall, performance scores broken down by quartiles (i.e,. -cq True), and performance scores for each individual label (i.e., -ci True). The directory scores will contain another directory called "analysis" containing the raw energy scores (i.e., -as True).


## Project members

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Christoph Metzner](https://github.com/cmetzner93) |     @cmetzner93    |







