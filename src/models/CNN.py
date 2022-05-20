"""
Source code that contains a basic convolution neural network architecture with three parallel convolution layers.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/20/2022
"""

# built-in libraries
from typing import List, Tuple, Union

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.attention_mechanisms import Attention


class CNN(nn.Module):
    """
    Convolution neural network class with three parallel convolution layers.

    Parameters
    ----------
    n_labels : int
        Number of labels considered in the label space
    n_cats : int
        Number of high-level categories to perform hierarchical attention
    embedding_dim : int
        Dimension of word embeddings, i.e., dense vector representation
    embedding_matrix : np.array
        Pre-trained embedding matrix using gensim's Word2Vec implementation
    window_sizes : List[int]; default = [3, 4, 5]
        Number of consecutive words considered in each convolution layer
    n_filters : List[int]
        Number of filters in each convolution layer
    dropout_p : float
        Probability of dropout layer
    att_module : str; default=None
        Defines the attention module/mechanism applied to perform attention to the latent document representation input
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled by the root of
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    cat_embedding_matrix : np.array
        Embedding matrix pretrained on the category descriptions

    """
    def __init__(self,
                 n_labels: int,
                 embedding_dim: int,
                 embedding_matrix: np.array = None,
                 window_sizes: List[int] = [3, 4, 5],
                 n_filters: List[int] = [500, 500, 500],
                 dropout_p: float = 0.5,
                 att_module: str = 'target',
                 scale: bool = False,
                 multihead: bool = False,
                 num_heads: int = None,
                 n_cats: int = None,
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None,
                 code2cat_map: List[int] = None):

        super().__init__()
        self._n_labels = n_labels
        self._n_cats = n_cats
        self._embedding_dim = embedding_dim
        self._window_sizes = window_sizes
        self._n_filters = n_filters
        self._dropout_p = dropout_p
        self._att_module = att_module
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map

        # Check to make sure window_sizes has same number of entries as num_filters
        if len(self._window_sizes) != len(self._n_filters):
            raise Exception("window_sizes must be same length as num_filters")

        # Init word embedding layer
        if embedding_matrix is None:
            self.embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=self._embedding_dim, padding_idx=0)
            self.embedding_layer.weight[0].data.fill_(0)  # set embedding layer weights of index 0 to 0
        else:
            embedding_matrix -= embedding_matrix.mean()
            embedding_matrix /= (embedding_matrix.std() * 20)
            self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),
                                                                freeze=False,
                                                                padding_idx=0)
        self.embedding_layer.weight[0].data.fill_(0)  # set embedding layer weights of index 0 to 0

        # Init dropout layer
        self.dropout_layer = nn.Dropout(p=self._dropout_p)

        # Init parallel convolution layers
        self.conv_layers = nn.ModuleList()
        for window_size, n_filter in zip(self._window_sizes, self._n_filters):
            conv_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                   out_channels=n_filter,
                                   padding='same',
                                   bias=True,
                                   kernel_size=window_size)
            nn.init.xavier_uniform_(conv_layer.weight)
            conv_layer.bias.data.fill_(0.01)
            self.conv_layers.append(conv_layer)

        # Init Attention Layer
        self.attention_layer = Attention(num_labels=self._n_labels,
                                         embedding_dim=self._embedding_dim,
                                         latent_doc_dim=np.sum(self._n_filters),
                                         att_module=self._att_module,
                                         scale=self._scale,
                                         multihead=self._multihead,
                                         num_heads=self._num_heads,
                                         num_cats=self._n_cats,
                                         label_embedding_matrix=self._label_embedding_matrix,
                                         cat_embedding_matrix=self._cat_embedding_matrix,
                                         code2cat_map=self._code2cat_map)

        # Init output layer
        self.output_layer = nn.Linear(in_features=np.sum(self._n_filters),
                                      out_features=self._n_labels)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, docs: torch.Tensor, return_doc_embeds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Creates a mask for words with boolean expression: True=word; False=padding
        mask_words = (docs != 0)
        words_per_line = mask_words.sum(-1)  # checks number of words for each line
        max_words = words_per_line.max()  # gets maximum number of words per doc

        # if statement to have line input of atleast 5 words due to conv layer with window size of 5
        max_window_size = max(self._window_sizes)
        if max_words < max_window_size:
            max_words = max_window_size

        # torch.unsqueeze required to reorder True/False mask to have the same shape as output of embedding layer
        mask_words = torch.unsqueeze(mask_words[:, :max_words], -1)
        docs_input_reduced = docs[:, :max_words]  # remove unnecessary padding

        # Generate word embeddings
        word_embeds = self.embedding_layer(docs_input_reduced)
        # torch.mul required to set created vector embeddings of padding token again to 0.
        word_embeds = torch.mul(word_embeds, mask_words.type(word_embeds.dtype))
        # Embedding layer output: (batch_size, sequence_length, embedding_size)
        # conv1d layer input: (batch_size, embedding_size, sequence_length)
        word_embeds = word_embeds.permute(0, 2, 1)

        # parallel 1D word convolutions
        conv_outs = []
        for layer in self.conv_layers:
            conv_out = F.relu(layer(word_embeds))
            conv_outs.append(conv_out)
        concat = torch.cat(conv_outs, 1)

        # Compute document embeddings contained in matrix H
        H = self.dropout_layer(concat)

        # Add attention module here
        C, att_scores = self.attention_layer(H=H)

        if self._att_module == 'self':
            # Necessary to match output with |L| ground-truth labels
            logits = self.output_layer(C).sum(dim=1)
        else:
            logits = self.output_layer(C).sum(dim=2)  # Consider .sum(dim=1) - depends on number of attention vectors

        if return_doc_embeds:
            return logits, H
        return logits
