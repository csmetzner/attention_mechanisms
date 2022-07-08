"""
This file contains source code of the dataloaders implemented in pytorch for the cancer pathology reports (PathReports)
and the clinical notes of MIMIC-III (Mimic) datasets.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/06/2022
    @last modified: 05/24/2022
"""

# Built-in libraries
from typing import Union, List, Dict

# Installed libraries
import numpy as np
import torch
from torch.utils.data import Dataset


class MimicData(Dataset):
    def __init__(self,
                 X: np.array,
                 Y: np.array,
                 transformer: bool = False,
                 doc_max_len: int = 4000):

        self.X = X
        self.Y = Y
        self._transformer = transformer
        self._doc_max_len = doc_max_len

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if self._transformer:
            sample = {'input_ids': self.X['input_ids'][idx],
                      'attention_mask': self.X['attention_mask'][idx]}
            sample['labels'] = torch.tensor(self.Y[idx], dtype=torch.float)
        else:
            doc = self.X[idx]  # get sample at idx from pd dataframe
            array = np.zeros(self._doc_max_len)  # create empty array filled with 0; 0 used for padding
            doc = doc[:self._doc_max_len]  # shorten document to max length
            doc_len = len(doc)  # get length of document
            array[:doc_len] = doc  # add document to empty array, if document shorter than max length then add 0-padding
            sample = {'X': torch.tensor(array, dtype=torch.long),
                      'Y': torch.tensor(self.Y[idx], dtype=torch.float)}  # create dict for current sample at idx
        return sample
