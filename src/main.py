"""
This file contains source code to run the experiments. Running the experiments includes loading the datasets, fitting
the models, and performing training, validating, and testing of the models.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/24/2022
"""

# built-in libraries
import os
import sys
import yaml
import pickle
import random
import argparse
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

print('Load libraries!', flush=True)
# installed libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# custom libraries
from tools.utils import parse_boolean, get_word_embedding_matrix
from tools import dataloaders
from tools.training import train, scoring
from models.CNN import CNN
from models.RNN import RNN
from models.Transformers import TransformerModel

# get root path
try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    root = os.path.dirname(os.getcwd())
sys.path.append(root)
print('Done!', flush=True)

# Two datasets: | PathReports | Mimic |
# - SEER cancer pathology reports  - Multiclass text classification
# - Physionet MIMIC-III  - Multilabel text classification

# Pytorch set device to 'cuda'/GPUs if available otherwise use available CPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The experiment uses the following device: {device}', flush=True)


class ExperimentSuite:
    def __init__(self,
                 model,
                 att_module,
                 dataset,
                 seed):
        self._dataset = dataset
        self._model = model
        self._att_module = att_module
        self.seed = seed
        self._model_args = None

        if self._model == 'DischargeBERT':
            self._transformer = True
        else:
            self._transformer = False

    def fetch_data(self):
        # Initialize list with split names
        splits = ['train', 'val', 'test']
        # Init path to load preprocessed data
        path_dataset = os.path.join(root, 'data', 'processed', f'data_{self._dataset}')

        # Initialize lists to store training, validation, and testing data
        # X: documents, Y: ground-truth labels
        X = []
        Y = []

        for s, split in enumerate(splits):
            # Load token2idx mapped documents
            if self._transformer:
                X_split = pd.read_pickle(os.path.join(path_dataset, f'X_{self._dataset}_{split}_text.pkl'))
                #Y_split = pd.read_pickle(os.path.join(path_dataset, f'y_code_{self._dataset}_{split}.pkl'))
                #Y_tensor = torch.stack([torch.from_numpy(sample) for sample in Y_split.values])
                #X_split['labels'] = Y_tensor
                X.append(X_split)
            else:
                X_split = pd.read_pickle(os.path.join(path_dataset, f'X_{self._dataset}_{split}.pkl'))
                X.append(X_split.values)

            # Load ground-truth values
            Y_split = pd.read_pickle(os.path.join(path_dataset, f'y_code_{self._dataset}_{split}.pkl'))
            Y.append(Y_split.values)
        return X, Y

    def fill_model_config(self,
                          model: str,
                          dataset: str,
                          att_module: str,
                          task: str,
                          embedding_dim: int = None,
                          dropout_p: float = None,
                          batch_size: int = None,
                          epochs: int = None,
                          doc_max_len: int = None,
                          patience: int = None,
                          scale: bool = False,
                          multihead: bool = False,
                          num_heads: int = None,
                          hidden_dim: int = None,
                          window_sizes: List[int] = None,
                          quartiles: bool = None,
                          individual: bool = None,
                          embedding_scaling: float = None) -> Dict[str, Union[str, Dict[str, Union[None, int, float, str, List[int]]]]]:

        # Create path to config_files
        path_config = os.path.join(root, 'src', 'config_files')
        path_data = os.path.join(root, 'data', 'processed')

        # Load config file for current model
        with open(os.path.join(path_config, f'{model}_config.yml'), 'r') as f:
            self._model_args = yaml.safe_load(stream=f)

        # Load config file for selected dataset
        with open(os.path.join(path_config, 'datasets_config.yml'), 'r') as f:
            datasets_config = yaml.safe_load(stream=f)

        # Retrieve required model arguments
        if dataset == 'PathReports':
            self._model_args['model_kwargs']['n_labels'] = datasets_config[dataset]['n_labels'][task]
            if att_module.split('_')[0] == 'hierarchical':
                self._model_args['model_kwargs']['n_cats'] = datasets_config[dataset]['n_cats'][task]
        else:  # Else: 'Mimic50' or 'MimicFull'
            self._model_args['model_kwargs']['n_labels'] = datasets_config[dataset]['n_labels']
            if att_module.split('_')[0] == 'hierarchical':
                self._model_args['model_kwargs']['n_cats'] = datasets_config[dataset]['n_cats']

        # Set max document length based on model - transformer model can only process 512 tokens
        if self._transformer:
            self._model_args['train_kwargs']['doc_max_len'] = 512
        else:
            self._model_args['train_kwargs']['doc_max_len'] = datasets_config[dataset]['doc_max_len']

        # Set attention mechanism
        self._model_args['model_kwargs']['att_module'] = att_module

        # Check if optional arguments were passed - if so, set to given value
        # Embedding Dimension:
        if embedding_dim is not None:
            self._model_args['model_kwargs']['embedding_dim'] = embedding_dim
        # Retrieve token embedding matrix
        token_embedding_matrix = get_word_embedding_matrix(dataset=dataset,
                                                           embedding_dim=self._model_args['model_kwargs']['embedding_dim'],
                                                           path_data=path_data,
                                                           min_count=3)

        # Add token embedding matrix to model args dict if not transformer - huggingface transformer models alread have
        if not self._transformer:
            self._model_args['model_kwargs']['token_embedding_matrix'] = token_embedding_matrix

        # Load pre-trained label and category embedding matrices, and label2category-mapping
        # Load mapping if hierarchical attention:
        if att_module.split('_')[0] == 'hierarchical':
            with open(os.path.join(path_data, 'code_embeddings', f'embedding_matrix_{dataset}_mapping.pkl'), 'rb') as f:
                code2cat_map = pickle.load(f)
            self._model_args['model_kwargs']['code2cat_map'] = code2cat_map

        if att_module == 'label':
            with open(os.path.join(path_data, 'code_embeddings', f'code_embedding_matrix_{dataset}_{self._model_args["model_kwargs"]["embedding_dim"]}.pkl'), 'rb') as f:
                label_embedding_matrix = pickle.load(f)
            self._model_args['model_kwargs']['label_embedding_matrix'] = label_embedding_matrix

        if att_module == 'hierarchical_label':
            with open(os.path.join(path_data, 'code_embeddings', f'code_embedding_matrix_{dataset}_{self._model_args["model_kwargs"]["embedding_dim"]}.pkl'), 'rb') as f:
                label_embedding_matrix = pickle.load(f)
            self._model_args['model_kwargs']['label_embedding_matrix'] = label_embedding_matrix
            with open(os.path.join(path_data, 'code_embeddings', f'cat_embedding_matrix_{dataset}_{self._model_args["model_kwargs"]["embedding_dim"]}.pkl'), 'rb') as f:
                cat_embedding_matrix = pickle.load(f)
            self._model_args['model_kwargs']['cat_embedding_matrix'] = cat_embedding_matrix

        # Add new value to model_args if given via commandline
        if dropout_p is not None:
            self._model_args['model_kwargs']['dropout_p'] = dropout_p
        if scale is not None:
            self._model_args['model_kwargs']['scale'] = scale
        if multihead is not None:
            self._model_args['model_kwargs']['multihead'] = multihead
        if num_heads is not None:
            self._model_args['model_kwargs']['num_heads'] = num_heads
        if batch_size is not None:
            self._model_args['train_kwargs']['batch_size'] = batch_size
        if epochs is not None:
            self._model_args['train_kwargs']['epochs'] = epochs
        if doc_max_len is not None:
            self._model_args['train_kwargs']['doc_max_len'] = doc_max_len
        if patience is not None:
            self._model_args['train_kwargs']['patience'] = patience
        if hidden_dim is not None:
            if self._model == 'CNN':
                self._model_args['model_kwargs']['n_filters'] = [int(hidden_dim)] * 3
            else:
                self._model_args['model_kwargs']['hidden_size'] = int(hidden_dim)
        if window_sizes is not None:
            if self._model == 'CNN':
                self._model_args['model_kwargs']['window_sizes'] = list(map(int, window_sizes))
            else:
                raise ValueError('Selected model has no window_sizes as model kwargs')
        if quartiles is not None:
            self._model_args['train_kwargs']['quartiles'] = quartiles
        if individual is not None:
            self._model_args['train_kwargs']['individual'] = individual
        if embedding_scaling is not None:
            self._model_args['model_kwargs']['embedding_scaling'] = embedding_scaling
        return self._model_args

    def fit_model(self,
                  model_args: Dict[str, Union[str, Dict[str, Union[str, int, List[Union[int, float]]]]]],
                  X: List[np.array],
                  Y: List[np.array],
                  path_res_dir: str):

        # Retrieve train kwargs
        doc_max_len = model_args['train_kwargs']['doc_max_len']
        batch_size = model_args['train_kwargs']['batch_size']
        lr = model_args['train_kwargs']['lr']

        # Check for required computation of performance metrics for quartiles and/or each individual labels
        quartiles = model_args['train_kwargs']['quartiles']
        individual = model_args['train_kwargs']['individual']

        # alignment = model_args['train_kwargs']['alignment']

        model_name = f"{self._model}" \
                     f"_{self._dataset}" \
                     f"_{model_args['train_kwargs']['batch_size']}" \
                     f"_{model_args['train_kwargs']['patience']}" \
                     f"_{model_args['model_kwargs']['att_module']}" \
                     f"_{model_args['model_kwargs']['embedding_dim']}" \
                     f"_{model_args['model_kwargs']['n_filters'][0] if self._model == 'CNN' else model_args['model_kwargs']['hidden_size']}" \
                     f"_{model_args['model_kwargs']['dropout_p']}" \
                     f"_{model_args['model_kwargs']['scale']}" \
                     f"_{model_args['model_kwargs']['multihead']}" \
                     f"_{model_args['model_kwargs']['num_heads']}" \
                     f'{timestamp}'

        print(f'Name of model: {model_name}')

        # Create directory to store model parameters
        path_res_models = os.path.join(path_res_dir, 'models/')
        if not os.path.exists(os.path.dirname(path_res_models)):
            os.makedirs(os.path.dirname(path_res_models))

        save_name = os.path.join(path_res_models, model_name)  # create absolute path to storage location

        Data = dataloaders.MimicData

        #if self._transformer:
        #    train_dataset = X[0]
        #    val_dataset = X[1]
        #    test_dataset = X[2]
        #    print(f'Size of training data {len(train_dataset["input_ids"])},'
        #          f' validation data {len(val_dataset["input_ids"])},'
        #          f' and testing data {len(test_dataset["input_ids"])}.')
        #else:
        # Training dataset
        train_dataset = Data(X=X[0], Y=Y[0], transformer=self._transformer, doc_max_len=doc_max_len)

        # Validation dataset
        val_dataset = Data(X=X[1], Y=Y[1], transformer=self._transformer, doc_max_len=doc_max_len)

        # Testing dataset
        test_dataset = Data(X=X[2], Y=Y[2], transformer=self._transformer, doc_max_len=doc_max_len)

        print(f'Size of training data {len(train_dataset)}, validation data {len(val_dataset)},'
              f' and testing data {len(test_dataset)}.', flush=True)

        # Setup pytorch DataLoader objects with training, validation, and testing dataset. Set shuffle to True for train
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if self._model == 'CNN':
            model = CNN(**model_args['model_kwargs'])
        elif self._model == 'BiLSTM':
            model = RNN(**model_args['model_kwargs'])
        elif self._model == 'LSTM':
            model = RNN(**model_args['model_kwargs'])
        elif self._model == 'BiGRU':
            model = RNN(**model_args['model_kwargs'])
        elif self._model == 'GRU':
            model = RNN(**model_args['model_kwargs'])
        elif self._model == 'DischargeBERT':
            model = TransformerModel(**model_args['model_kwargs'])
        else:
            raise Exception('Invalid model type!')

        # Set up parallel computing if possible
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

        train(model=model,
              train_kwargs=model_args['train_kwargs'],
              optimizer=optimizer,
              train_loader=train_loader,
              transformer=self._transformer,
              val_loader=val_loader,
              class_weights=None,
              save_name=save_name)

        # Test the best model - load it
        model.load_state_dict(torch.load(os.path.join(f'{save_name}.pt')))
        model.to(device)

        # Save the test_scores to csv file
        print('Testing trained model')

        # Check if performance metrics should be computed for quartiles
        if quartiles:
            with open(os.path.join(root, 'data', 'processed', f'data_{self._dataset}', f'l_codes_quantiles_{self._dataset}.pkl'), 'rb') as f:
                quartiles_indices = pickle.load(f)
        else:
            quartiles_indices = None

        test_scores = scoring(model=model,
                              data_loader=test_loader,
                              transformer=self._transformer,
                              class_weights=None,
                              quartiles_indices=quartiles_indices,
                              individual=individual)

        print(f'Test loss: {test_scores["loss"]}', flush=True)

        store_scores(scores=test_scores,
                     model_type=self._model,
                     dataset=self._dataset,
                     seed= self.seed,
                     train_kwargs=model_args['train_kwargs'],
                     model_kwargs=model_args['model_kwargs'],
                     path_res_dir=path_res_dir,
                     model_name=model_name,
                     quartiles=quartiles,
                     individual=individual)


def store_scores(scores: Dict[str, Union[List[float], float]],
                 model_type: str,
                 dataset: str,
                 seed: int,
                 train_kwargs: Dict[str, int],
                 model_kwargs: Dict[str, Union[int, str]],
                 path_res_dir: str,
                 model_name: str,
                 quartiles: bool = False,
                 individual: bool = False):

    # Create results directories: predictions, models, scores
    path_res_preds = os.path.join(path_res_dir, 'predictions/')
    path_res_scores = os.path.join(path_res_dir, 'scores/')
    if not os.path.exists(os.path.dirname(path_res_preds)):
        os.makedirs(os.path.dirname(path_res_preds))
    if not os.path.exists(os.path.dirname(path_res_scores)):
        os.makedirs(os.path.dirname(path_res_scores))

    metrics = ['f1_macro_sk', 'f1_micro_sk',
               'auc_micro', 'auc_macro', 'prec@5', 'prec@8', 'prec@15']
    columns = ['dataset',
               'seed'
               'doc_max_len',
               'batch_size',
               'patience',
               'att_module',
               'word_embedding_dim',
               'kernel_sizes',
               'hidden_dim',
               'dropout_p',
               'scale',
               'multihead',
               'num_heads',
               'f1_macro_sk', 'f1_micro_sk', 'auc_micro', 'auc_macro', 'prec@5', 'prec@8', 'prec@15']

    file_name = f'scores.xlsx'
    path_save_xlsx = os.path.join(path_res_scores, file_name)
    scores_to_excel = {f'{model_type}': [dataset,
                                         seed,
                                         train_kwargs['doc_max_len'],
                                         train_kwargs['batch_size'],
                                         train_kwargs['patience'],
                                         model_kwargs['att_module'],
                                         model_kwargs['embedding_dim'],
                                         model_kwargs['window_sizes'] if model_type == 'CNN' else 'none',
                                         model_kwargs['n_filters'] if model_type == 'CNN' else model_kwargs['hidden_size'],
                                         model_kwargs['dropout_p'],
                                         model_kwargs['scale'],
                                         model_kwargs['multihead'],
                                         model_kwargs['num_heads']]}

    for metric in metrics:
        value = scores[metric]
        scores_to_excel[f'{model_type}'].append(value)

    df = pd.DataFrame.from_dict(data=scores_to_excel, orient='index', columns=columns)

    if os.path.isfile(path=path_save_xlsx):
        writer = pd.ExcelWriter(path=path_save_xlsx, engine='openpyxl', mode='a', if_sheet_exists='overlay')
        df.to_excel(excel_writer=writer,
                    sheet_name='Sheet1',
                    index=True,
                    float_format="%.3f",
                    na_rep='NaN',
                    startrow=writer.sheets['Sheet1'].max_row,
                    header=None)
    else:
        writer = pd.ExcelWriter(path=path_save_xlsx, engine='xlsxwriter', mode='w')
        df.to_excel(excel_writer=writer,
                    sheet_name='Sheet1',
                    index=True,
                    index_label='Model',
                    float_format="%.3f",
                    na_rep='NaN')

    writer.save()

    # Retrieve predictions
    ids = pd.read_csv(os.path.join(root, 'data', 'processed', f'data_{dataset}', f'ids_{dataset}_test.csv'))
    with open(os.path.join(root, 'data', 'processed', f'data_{dataset}', f'l_codes_{dataset}.pkl'), "rb") as f:
        class_names = pickle.load(f)

    with open(os.path.join(path_res_preds, f'{model_name}_test.txt'), 'w') as file:
        for hadm_id, y_pred_doc in zip(ids.HADM_ID.tolist(), scores['y_preds']):
            row = f'{hadm_id}'
            for label, y_pred in zip(class_names, y_pred_doc):
                if y_pred == 1:
                    row += f'|{label}'
            row += '\n'
            file.write(row)

    if quartiles:
        metrics = []
        columns = ['dataset',
                   'seed',
                   'doc_max_len',
                   'batch_size',
                   'patience',
                   'att_module',
                   'word_embedding_dim',
                   'kernel_sizes',
                   'hidden_dim',
                   'dropout_p',
                   'scale',
                   'multihead',
                   'num_heads']

        for quartile_idx in range(4):
            metrics.append(f'f1_macro_Q{quartile_idx}')
            metrics.append(f'f1_micro_Q{quartile_idx}')
            columns.append(f'f1_macro_Q{quartile_idx}')
            columns.append(f'f1_micro_Q{quartile_idx}')

        file_name = f'scores_quartiles_{dataset}.xlsx'

        path_save_xlsx = os.path.join(path_res_scores, file_name)
        scores_to_excel = {f'{model_type}': [dataset,
                                             seed,
                                             train_kwargs['doc_max_len'],
                                             train_kwargs['batch_size'],
                                             train_kwargs['patience'],
                                             model_kwargs['att_module'],
                                             model_kwargs['embedding_dim'],
                                             model_kwargs['window_sizes'] if model_type == 'CNN' else 'none',
                                             model_kwargs['n_filters'] if model_type == 'CNN' else model_kwargs[
                                                 'hidden_size'],
                                             model_kwargs['dropout_p'],
                                             model_kwargs['scale'],
                                             model_kwargs['multihead'],
                                             model_kwargs['num_heads']]}

        for metric in metrics:
            value = scores[metric]
            scores_to_excel[f'{model_type}'].append(value)

        df = pd.DataFrame.from_dict(data=scores_to_excel, orient='index', columns=columns)

        if os.path.isfile(path=path_save_xlsx):
            writer = pd.ExcelWriter(path=path_save_xlsx, engine='openpyxl', mode='a', if_sheet_exists='overlay')
            df.to_excel(excel_writer=writer,
                        sheet_name='Sheet1',
                        index=True,
                        float_format="%.3f",
                        na_rep='NaN',
                        startrow=writer.sheets['Sheet1'].max_row,
                        header=None)
        else:
            writer = pd.ExcelWriter(path=path_save_xlsx, engine='xlsxwriter', mode='w')
            df.to_excel(excel_writer=writer,
                        sheet_name='Sheet1',
                        index=True,
                        index_label='Model',
                        float_format="%.3f",
                        na_rep='NaN')

        writer.save()

    if individual:
        metrics = []
        columns = ['dataset',
                   'seed',
                   'doc_max_len',
                   'batch_size',
                   'patience',
                   'att_module',
                   'word_embedding_dim',
                   'kernel_sizes',
                   'hidden_dim',
                   'dropout_p',
                   'scale',
                   'multihead',
                   'num_heads']

        for i, label in enumerate(class_names):
            metrics.append(f'f1_micro_label{i}')
            columns.append(f'f1_micro_{label}')

        file_name = f'scores_individual_{dataset}.xlsx'

        path_save_xlsx = os.path.join(path_res_scores, file_name)
        scores_to_excel = {f'{model_type}': [dataset,
                                             seed,
                                             train_kwargs['doc_max_len'],
                                             train_kwargs['batch_size'],
                                             train_kwargs['patience'],
                                             model_kwargs['att_module'],
                                             model_kwargs['embedding_dim'],
                                             model_kwargs['window_sizes'] if model_type == 'CNN' else 'none',
                                             model_kwargs['n_filters'] if model_type == 'CNN' else model_kwargs[
                                                 'hidden_size'],
                                             model_kwargs['dropout_p'],
                                             model_kwargs['scale'],
                                             model_kwargs['multihead'],
                                             model_kwargs['num_heads']]}

        for metric in metrics:
            value = scores[metric]
            scores_to_excel[f'{model_type}'].append(value)

        df = pd.DataFrame.from_dict(data=scores_to_excel, orient='index', columns=columns)

        if os.path.isfile(path=path_save_xlsx):
            writer = pd.ExcelWriter(path=path_save_xlsx, engine='openpyxl', mode='a', if_sheet_exists='overlay')
            df.to_excel(excel_writer=writer,
                        sheet_name='Sheet1',
                        index=True,
                        float_format="%.3f",
                        na_rep='NaN',
                        startrow=writer.sheets['Sheet1'].max_row,
                        header=None)
        else:
            writer = pd.ExcelWriter(path=path_save_xlsx, engine='xlsxwriter', mode='w')
            df.to_excel(excel_writer=writer,
                        sheet_name='Sheet1',
                        index=True,
                        index_label='Model',
                        float_format="%.3f",
                        na_rep='NaN')

        writer.save()


# Use argparse library to set up command line arguments
parser = argparse.ArgumentParser()
# Model-unspecific commandline arguments
parser.add_argument('-m', '--model',
                    required=True,
                    type=str,
                    choices=['CNN', 'LSTM', 'BiLSTM', 'GRU', 'BiGRU', 'DischargeBERT'],
                    help='Select a predefined model.')
parser.add_argument('-d', '--dataset',
                    required=True,
                    type=str,
                    choices=['PathReports', 'MimicFull', 'Mimic50'],
                    help='Select a preprocessed dataset.')
parser.add_argument('-am', '--attention_module',
                    required=True,
                    type=str,
                    choices=['target', 'self', 'label', 'alternate', 'hierarchical_target', 'hierarchical_label',
                             'hierarchical_context', 'hierarchical_double_attention',
                             'context', 'context_diff',
                             'max_masked', 'rank_masked'],
                    help='Select a type of predefined attention mechanism or none.'
                         '-none: No Attention'
                         '-target: Target Attention')
parser.add_argument('-en', '--experiment_name',
                    required=True,
                    type=str,
                    help='Set name of experiment')
parser.add_argument('-t', '--task',
                    type=str,
                    choices=['site', 'subsite', 'laterality', 'grade', 'histology', 'behavior'],
                    help='Select a task if dataset "PathReports" was selected.'
                         '\nTasks: | site | subsite | laterality | grade | histology | behavior |')
parser.add_argument('-ed', '--embedding_dim',
                    type=int,
                    help='Set embedding dimension; optional.'
                         '\nIf pretrained embedding matrix does not exist for dim then is created.')
parser.add_argument('-sca', '--scale',
                    type=parse_boolean,
                    help='Set flag if energy scores should be scores: E = (QxK.T/root(emb_dim)).')
parser.add_argument('-mh', '--multihead',
                    type=parse_boolean,
                    help='Flag indicating if multihead attention mechanism should be used.')
parser.add_argument('-nh', '--num_heads',
                    type=int,
                    help='Set number of attention heads if multihead argument is set to True.')
parser.add_argument('-dp', '--dropout_p',
                    type=float,
                    help='Set dropout probability; optional.')
parser.add_argument('-e', '--epochs',
                    type=int,
                    help='Set number of epochs; optional.')
parser.add_argument('-op', '--optimizer',
                    choices=['Adam', 'AdamW'],
                    help='Set optimizer for training; optional.')
parser.add_argument('-dml', '--doc_max_len',
                    type=int,
                    help='Set maximum document length; optional.')
parser.add_argument('-bs', '--batch_size',
                    type=int,
                    help='Set batch size; optional.')
parser.add_argument('-p', '--patience',
                    type=int,
                    help='Set patience value (until early stopping actives); optional.')
parser.add_argument('-sc', '--singularity',
                    default=False,
                    type=parse_boolean,
                    help='Set flag to store results in directory binded between container and home directory.')
parser.add_argument('-hd', '--hidden_dim',
                    help='Set hidden dimension.')
parser.add_argument('-ws', '--window_sizes',
                    nargs='+',
                    help='Set window sizes for the 3 conv layers in CNN model (e.g., -ws 3 4 5)')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help='Set seed number for reproducibility.')
parser.add_argument('-cq', '--compute_quartiles',
                    type=parse_boolean,
                    help='Compute performance metrics based on quartile splits of label set.')
parser.add_argument('-ci', '--compute_individual',
                    type=parse_boolean,
                    help='Compute individual label performance metrics.')
parser.add_argument('-es', '--embedding_scaling',
                    type=float,
                    help='Set scaler for token/label/category embedding normalization')

args = parser.parse_args()


def main():
    # Select seed for reproducibility
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    exp = ExperimentSuite(model=args.model,
                          att_module=args.attention_module,
                          dataset=args.dataset,
                          seed=SEED)

    if (args.dataset == 'PathReports') and (args.task is None):
        raise TypeError('If dataset "PathReports" is selected, you MUST select a task.'
                        '\nTasks: | site | subsite | laterality | grade | histology | behavior |')

    if args.multihead and (args.num_heads is None):
        raise TypeError('Enter number of attention heads when multihead is set to True.'
                        '\n| -nh | --num_heads |')

    model_args = exp.fill_model_config(model=args.model,
                                       dataset=args.dataset,
                                       att_module=args.attention_module,
                                       task=args.task,
                                       embedding_dim=args.embedding_dim,
                                       dropout_p=args.dropout_p,
                                       batch_size=args.batch_size,
                                       epochs=args.epochs,
                                       doc_max_len=args.doc_max_len,
                                       patience=args.patience,
                                       scale=args.scale,
                                       multihead=args.multihead,
                                       num_heads=args.num_heads,
                                       hidden_dim=args.hidden_dim,
                                       window_sizes=args.window_sizes,
                                       quartiles=args.compute_quartiles,
                                       individual=args.compute_individual,
                                       embedding_scaling=args.embedding_scaling)

    if args.singularity:
        path_res_dir = f'mnt/results_{args.experiment_name}/'
    else:
        path_res_dir = os.path.join(root, f'results_{args.experiment_name}')

    if not os.path.exists(os.path.dirname(path_res_dir)):
        os.makedirs(os.path.dirname(path_res_dir))

    X, Y = exp.fetch_data()

    exp.fit_model(model_args=model_args,
                  X=X,
                  Y=Y,
                  path_res_dir=path_res_dir)


if __name__ == '__main__':
    main()
