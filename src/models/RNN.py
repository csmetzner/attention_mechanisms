"""
This file contains source code for the pytorch implementation of different recurrent neural network types such as the
Long Short-term Memory Model (LSTM) or the Gated Recurrent Unit Model (GRU) for one- or bi-directional.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/13/2022
"""

# built-in libraries
from typing import Tuple, Union

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.attention_mechanisms import TargetAttention, SelfAttention, LabelAttention, AlternateAttention
from attention_modules.attention_mechanisms import HierarchicalTargetAttention, HierarchicalLabelAttention


class RNN(nn.Module):
    def __init__(self,
                 n_labels: int,
                 n_cats: int,
                 embedding_matrix: np.array,
                 embedding_dim: int,
                 hidden_size: int,
                 RNN_type: str,
                 bidir: bool = True,
                 n_layers: int = 2,
                 dropout_p: float = 0.5,
                 att_module: str = 'target',
                 scale: bool = False,
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None):
        super().__init__()
        self._n_labels = n_labels
        self._n_cats = n_cats
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._RNN_type = RNN_type
        self._bidir = bidir
        self._n_layers = n_layers
        self._dropout_p = dropout_p
        self._att_module = att_module
        self._scale = scale
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix

        # Init word embedding layer
        embedding_matrix -= embedding_matrix.mean()
        embedding_matrix /= (embedding_matrix.std() * 20)
        self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),
                                                            freeze=False,
                                                            padding_idx=0)
        self.embedding_layer.weight[0].data.fill_(0)  # set embedding layer weights of index 0 to 0

        # Init dropout layer
        self.dropout_layer = nn.Dropout(p=self._dropout_p)

        # Init RNN type
        if self._RNN_type == 'LSTM':
            self._RNN = nn.LSTM(input_size=self._embedding_dim,
                                hidden_size=self._hidden_size,
                                num_layers=self._n_layers,
                                dropout=self._dropout_p if self._n_layers > 1 else 0,
                                bias=True,
                                batch_first=True,
                                bidirectional=self._bidir)
        elif self._RNN_type == 'GRU':
            self._RNN = nn.GRU(input_size=self._embedding_dim,
                               hidden_size=self._hidden_size,
                               num_layers=self._n_layers,
                               dropout=self._dropout_p if self._n_layers > 1 else 0,
                               bias=True,
                               batch_first=True,
                               bidirectional=self._bidir)

        # define attention layer based on attention module
        if self._bidir:
            self._hidden_size = self._hidden_size * 2

        if self._att_module == 'target':
            self.att_layer = TargetAttention(encoder_out_dim=self._hidden_size,
                                             n_att_vectors=self._n_labels,
                                             embedding_dim=self._embedding_dim,
                                             scale=self._scale)
        elif self._att_module == 'self':
            self.att_layer = SelfAttention(encoder_out_dim=self._hidden_size,
                                           embedding_dim=self._embedding_dim,
                                           scale=self._scale)
        elif self._att_module == 'label':
            self.att_layer = LabelAttention(encoder_out_dim=self._hidden_size,
                                            embedding_dim=self._embedding_dim,
                                            n_labels=self._n_labels,
                                            label_embedding_matrix=label_embedding_matrix,
                                            scale=self._scale)
        elif self._att_module == 'alternate':
            self.att_layer = AlternateAttention(encoder_out_dim=self._hidden_size,
                                                n_att_vectors=self._n_labels,
                                                scale=self._scale)
        elif self._att_module == 'hierarchical_target':
            self.att_layer = HierarchicalTargetAttention(encoder_out_dim=self._hidden_size,
                                                         n_labels_lvl_1=self._n_cats,
                                                         n_labels_lvl_2=self._n_labels,
                                                         scale=self._scale)
        elif self._att_module == 'hierarchical_label':
            self.att_layer = HierarchicalLabelAttention(encoder_out_dim=self._hidden_size,
                                                        embedding_dim=self._embedding_dim,
                                                        n_labels_lvl_1=self._n_cats,
                                                        n_labels_lvl_2=self._n_labels,
                                                        cat_embedding_matrix=self._cat_embedding_matrix,
                                                        label_embedding_matrix=self._label_embedding_matrix,
                                                        scale=self._scale)

        # Init output layer
        self.output_layer = nn.Linear(in_features=self._hidden_size,
                                      out_features=self._n_labels)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, docs: torch.Tensor, return_doc_embeds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Creates a mask for words with boolean expression: True=word; False=padding
        mask_words = (docs != 0)
        words_per_line = mask_words.sum(-1)  # checks number of words for each line
        max_words = words_per_line.max()  # gets maximum number of words per doc

        # torch.unsqueeze required to reorder True/False mask to have the same shape as output of embedding layer
        mask_words = torch.unsqueeze(mask_words[:, :max_words], -1)
        docs_input_reduced = docs[:, :max_words]  # remove unnecessary padding

        # Generate word embeddings
        word_embeds = self.embedding_layer(docs_input_reduced)
        # torch.mul required to set created vector embeddings of padding token again to 0.
        # Embedding layer output: (batch_size, sequence_length, embedding_size)er4w
        word_embeds = torch.mul(word_embeds, mask_words.type(word_embeds.dtype))

        # Compute document representations using the Bi-LSTM
        H = self._RNN(word_embeds)
        H = H[0].permute(0, 2, 1)

        # Add attention module here
        C, att_scores = self.att_layer(H=H)

        if self._att_module == 'self':
            # Necessary to match output with |L| ground-truth labels
            logits = self.output_layer(C).sum(dim=1)
        else:
            logits = self.output_layer(C).sum(dim=2)  # Consider .sum(dim=1) - depends on number of attention vectors

        if return_doc_embeds:
            return logits, H
        return logits
