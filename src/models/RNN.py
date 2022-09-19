"""
This file contains source code for the pytorch implementation of different recurrent neural network types such as the
Long Short-term Memory Model (LSTM) or the Gated Recurrent Unit Model (GRU) for one- or bi-directional.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/24/2022
"""

# built-in libraries
from typing import Tuple, Union, List

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.attention_mechanisms import Attention


class RNN(nn.Module):
    """
    Recurrent neural network class

    Parameters
    ----------
    n_labels : int
        Number of labels considered in the label space
    embedding_dim : int
        Dimension of word embeddings, i.e., dense vector representation
    embedding_matrix : np.array
        Pre-trained embedding matrix using gensim's Word2Vec implementation
    hidden_size : int
        Dimension of hidden states of RNN
    RNN_type : str
        Selected type of RNN: | LSTM | BiLSTM | GRU | BiGRU|
    bidir : bool; default=True
        Flag indicating if RNN is bi-directional.
    n_layers : int; default=2
        Number of layers of RNN
    dropout_p : float
        Probability of dropout layer
    att_module : str; default=None
        Defines the attention module/mechanism applied to perform attention to the latent document representation input
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled by the root of
    multihead : bool; default=False
        Flag indicating if multi-head attention is used
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated
    n_cats : int
        Number of high-level categories to perform hierarchical attention
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    cat_embedding_matrix : np.array
        Embedding matrix pretrained on the category descriptions
    code2cat_map : List[int]; default=None
        Category index to map codes to categories
    """
    def __init__(self,
                 n_labels: int,
                 embedding_dim: int,
                 token_embedding_matrix: np.array,
                 hidden_size: int,
                 RNN_type: str,
                 bidir: bool = True,
                 n_layers: int = 2,
                 dropout_p: float = 0.5,
                 att_module: str = 'target',
                 scale: bool = False,
                 multihead: int = False,
                 num_heads: int = None,
                 n_cats: int = None,
                 embedding_scaling: int = 1,
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None,
                 code2cat_map: List[int] = None):
        super().__init__()
        self._n_labels = n_labels
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._RNN_type = RNN_type
        self._bidir = bidir
        self._n_layers = n_layers
        self._dropout_p = dropout_p
        self._att_module = att_module
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self._n_cats = n_cats
        self._embedding_scaling = embedding_scaling
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map

        # Init word embedding layer
        token_embedding_matrix -= token_embedding_matrix.mean()
        token_embedding_matrix /= (token_embedding_matrix.std() * self._embedding_scaling)
        self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(token_embedding_matrix, dtype=torch.float),
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

        # Init Attention Layer
        if self._att_module != 'max_pool':
            self.attention_layer = Attention(num_labels=self._n_labels,
                                             embedding_dim=self._embedding_dim,
                                             latent_doc_dim=self._hidden_size,
                                             att_module=self._att_module,
                                             scale=self._scale,
                                             multihead=self._multihead,
                                             num_heads=self._num_heads,
                                             num_cats=self._n_cats,
                                             embedding_scaling=self._embedding_scaling,
                                             label_embedding_matrix=self._label_embedding_matrix,
                                             cat_embedding_matrix=self._cat_embedding_matrix,
                                             code2cat_map=self._code2cat_map)

        # Init output layer
        self.output_layer = nn.Linear(in_features=self._hidden_size,
                                      out_features=self._n_labels)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, docs: torch.Tensor, return_doc_embeds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass of RNN models

        Parameters
        ----------
        docs : torch.tensor
            Input documents
        return_doc_embeds : bool; default=False
            Flag indicating if doc embeddings should be returned

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor]]
            [Logits], [Logits, doc_embeds]


        """
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
        H = self.dropout_layer(H)

        # Add attention module here
        if self._att_module == 'max_pool':
            logits = self.output_layer(H.permute(0, 2, 1)).permute(0, 2, 1)
            logits = F.adaptive_max_pool1d(logits, 1)
            logits = torch.flatten(logits, start_dim=1)
        else:
            C, att_scores = self.attention_layer(H=H)
            logits = self.output_layer.weight.mul(C).sum(dim=2).add(self.output_layer.bias)
        return logits
