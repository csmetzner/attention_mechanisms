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
│
├──
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
│   ├── notebook_preprocessing_MIMIC_III_and_code_descriptions.ipynb      <- Contains code to preprocess the raw MIMIC-III clinical documents and ICD-9 code descriptions
│   └── notebook_results_Mimic_III.ipynb          <- Contains code used to visualize results
│
├── src
│   ├── attention_modules
│   ├── config_files
│   ├── models
│   ├── tools
│   ├── .gitkeep
│   ├── __init__.py
│   ├── main.py
│   ├── print_results.py
│   └──   
│
│
│
│
│
│

  

```

## Project members

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Christoph Metzner](https://github.com/cmetzner93) |     @cmetzner93    |







