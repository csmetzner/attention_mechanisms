"""
This file contains source code for utility functions for preprocessing the raw data.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/06/2022
    @last modified: 03/23/2023
"""

SEED = 42
# Built-in libraries
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")
from typing import List

# Installed libraries
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
tqdm.pandas()
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer", model_max_length=4096)


def rel2abs(x: str, flag_proc: bool = True) -> str:
    """ Function that transform relative ICD-9 code into absolute code

    Parameters
    ----------
    x : str
        relative ICD-9 code
    flag_proc : bool; default=True
        flag indicating if relative code is a procedure or diagnosis code

    Returns
    -------
    str
        absolute ICD-9 code, e.g., procedure: XX.XX / diagnosis: XXX.XX

    """
    if flag_proc:
        # Some codes are billable with only two or three digits, do not add period
        if len(x) == 2:
            return x
        else:
            return f'{x[:2]}.{x[2:]}'
    else:
        if len(x) == 3:
            return x
        else:
            if x[0] == 'E':
                if len(x) == 4:
                    return x
                else:
                    return f'{x[:4]}.{x[4:]}'
            else:
                return f'{x[:3]}.{x[3:]}'


def preproc_clinical_notes(df_notes: pd.DataFrame,
                           path_data_proc: str) -> pd.DataFrame:
    """
    Function that loads table NOTEEVENTS.csv of the MIMIC-III. This table contains all available clinical notes
    associated with each unique hospital admission id. The clinical notes are filtered for the discharge summaries. The
    category 'Discharge summary' contains the report and any addendum associated with the id.

    Parameters
    ----------
    df_notes : pd.DataFrame
        raw clinical notes from category 'Discharge summary'
    path_data_proc : str
        Path to directory where processed text should be stored

    Returns
    -------
    pd.DataFrame
        Dataframe that contains the concatenated clinical notes per hospital admission id.

    """
    # Merge reports and addendums from same patient/ham_id
    df_notes['TEXT'] = df_notes.groupby(['HADM_ID'])['TEXT'].transform(lambda x: '\n '.join(x))
    df_notes = df_notes.drop_duplicates(['HADM_ID']).copy()

    # Cleaning of clinical notes
    # Used for CNN and both RNNs
    df_notes['TOKENS'] = df_notes.progress_apply(lambda x: clean_tokenize_text(str(x.TEXT), tokenize=True), axis=1)
    # Used for the HuggingFace based Clinical Longformer
    df_notes['TEXT'] = df_notes.progress_apply(lambda x: clean_tokenize_text(str(x.TEXT), tokenize=False), axis=1)

    # Store dataframe
    df_notes.to_pickle(os.path.join(path_data_proc, 'CLEANED_NOTES.pkl'))

    return df_notes


def clean_tokenize_text(text: str, tokenize: bool = True) -> str:
    # define punctuations
    '''
    Patterns to remove
    [**Last Name **]
    [**First Name **]
    [**Name **]
    [**Date **]
    [**int-int **]
    [**Location **]
    [**MD **]
    [**Job **]
    [**Doctor **]
    [**Hospital **]
    [**Telephone **]
    [** **]
    '''
    # 1. Remove identifiers
    text = re.sub('\[\*\*.*?\*\*\]', 'deidentified', text)  # remove identifiers
    # 2. lowercase words and remove numeric tokens
    tokens = [t.lower() for t in word_tokenize(text) if not t.isnumeric()]
    if tokenize:
        return tokens
    # 3. Huggingface tokenizer expects a string to be fed for tokenization
    text = ' '.join(tokens)
    return text


def get_class_type(classes_list: List[str], code: bool=True) -> List[List[str]]:
    """
    Helper function that helps extracting procedure and diagnosis codes from all classes included in one list for
    multitask learning. This extraction is required since procedure or diagnosis codes have different structure.
    Procedure: XX.XX / Diagnosis: XXX.XX

    Parameters
    ----------
    classes_list : List[str]
        List that contains all unique classes for the billable ICD-9 codes or ICD-9 categories. These lists contain
        procedure and diagnosis codes in one.
    code : bool
        Flag indicating whether the function is fed list with billable codes or categories.

    Returns
    -------
    List[List[str]]
        A list containing two lists that contains either procedure or diagnosis codes/categories.
    """
    classes_proc = []
    classes_diag = []
    if code:
        for x in classes_list:
            if '.' in x:
                if x[2] == '.':
                    classes_proc.append(x)
                elif (x[3] == '.') or (x[4] == '.'):
                    classes_diag.append(x)
            else:
                if len(x) == 2:
                    classes_proc.append(x)
                elif (len(x) == 3) or (len(x) == 4):
                    classes_diag.append(x)
    else:
        for x in classes_list:
            if len(x) == 2:
                classes_proc.append(x)
            elif (len(x) == 3) or (len(x) == 4):
                classes_diag.append(x)

    return [classes_proc, classes_diag]


def create_splits(subset: str, path_data_proc: str, min_freq: int = 3):
    """
    This function creates X and y data for training, testing, and validation splits for each subset.

    Parameters
    ----------
    subset : str
        Name of subset: 'full' | '50' |
    path_data_proc : str
        Path to directory where processed data is stored
    min_freq : int
        Minimum frequence of token occurring in training dataset to be considered, otherwise taken as unknown: <unk>

    """
    splits = ['train', 'test', 'val']

    # Load subset specific data (processed data and lists of labels - codes/cats)
    df = pd.read_pickle(os.path.join(path_data_proc, f'data_{subset}', f'DATA_{subset}.pkl'))
    with open(os.path.join(path_data_proc, f'data_{subset}', f'l_codes_{subset}.pkl'), 'rb') as f:
        classes_code = pickle.load(f)

    # Init sklearn multi-hot label encoder
    mlb_code = MultiLabelBinarizer()
    mlb_code.fit([classes_code])

    # Create vocabulary etc.
    print(f'\nCurrent subset: {subset}')
    for split in splits:
        print(f'Current split: {split}')
        # Retrieve HADM_ID published by Mullenbach et al. 2018 for subsets 'full' and '50' if CAML flag is True
        hadm_id = pd.read_csv(
            os.path.join(path_data_proc, 'hadm_ids', f'hadm_ids_{subset}_{split}.csv')).hadm_id.tolist()

        # Create dataframe for current split
        df_split = df[df['HADM_ID'].isin(hadm_id)]

        # and clinical notes (future tabular data) and labels (codes and categories)
        df_split_ids = df_split[['SUBJECT_ID', 'HADM_ID']].copy()
        df_split_ids.to_csv(os.path.join(path_data_proc, f'data_{subset}', f'ids_{subset}_{split}.csv'), index=False)

        ###################
        ## Create X data ##
        ###################
        # Store just the cleaned data to be used for huggingface/transformers
        df_split_X_text = df_split[['TEXT']].copy()  # make list of individual strings, each string is one sample
        token2id_transformer = tokenizer(df_split_X_text.TEXT.tolist(),
                                         padding=True,
                                         truncation=True,
                                         return_tensors='pt')

        with open(os.path.join(path_data_proc, f'data_{subset}', f'X_{subset}_{split}_text.pkl'), 'wb') as f:
            pickle.dump(token2id_transformer, f)

        # Map tokens2idx
        df_split_X = df_split[['TOKENS']].copy()

        # Create vocabulary from training split using torchtext class build_vocab_from_iterator
        if split == 'train':
            # Create list containing
            train_tokens = df_split_X.TOKENS.tolist()
            with open(os.path.join(path_data_proc, f'data_{subset}', f'train_tokens_{subset}.pkl'), 'wb') as f:
                pickle.dump(train_tokens, f)

            vocab = build_vocab_from_iterator(train_tokens, min_freq=min_freq, specials=['<pad>', '<unk>'])
            vocab.set_default_index(1)
            print(f'Vocab size for {split}: {len(vocab)}')

            with open(os.path.join(path_data_proc, f'data_{subset}', f'vocab_{subset}.pkl'), 'wb') as f:
                pickle.dump(vocab, f)

        # apply mapping of tokens to index function
        df_split_X['token2id'] = df_split_X.apply(lambda x: vocab.lookup_indices(x.TOKENS), axis=1)
        df_split_X.token2id.to_pickle(os.path.join(path_data_proc, f'data_{subset}', f'X_{subset}_{split}.pkl'))

        ###################
        ## Create y data ##
        ###################

        df_split_y_code = df_split[['ICD9_CODE']].copy()

        # Binarize ground-truth labels
        # billable codes
        df_split_y_code['y_code'] = df_split_y_code.apply(lambda x: mlb_code.transform([x.ICD9_CODE])[0], axis=1)

        # Store encoded ground-truth labels
        df_split_y_code.y_code.to_pickle(os.path.join(path_data_proc, f'data_{subset}', f'y_code_{subset}_{split}.pkl'))
