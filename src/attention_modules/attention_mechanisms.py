"""
This file contains source code contains the pytorch implementation for the different attention mechanisms. The scripts
are written to modularize the different variations.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/20/2022

Attention mechanisms:
    - Self-attention (implemented, tested)
    - Target-attention (implemented, tested)
    - Label-attention (implemented, tested)
    - Hierarchical-attention
        - Target attention (implemented, tested)
        - Label attention (implemented, tested)
    - Multi-head attention (implemented, tested; https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html)
    - Alternating attention (implemented, tested)
"""
# built-in libraries
from typing import Tuple, Union, List

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.multihead_attention import transpose_output
from attention_modules.random_attention import RandomAttention
from attention_modules.pretrained_attention import PretrainedAttention
from attention_modules.hierarchical_attention import HierarchicalRandomAttention, HierarchicalPretrainedAttention
from attention_modules.target_attention import TargetAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    """
    General attention class that initializes and performs selected attention mechanism.

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in low-level label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    att_module : str
        Selected attention mechanism
            target: Query matrix (Q) is randomly initialized
            label: Query matrix is initialized using sentence embedding of code descriptions of the label space
            self: Query matrix is the token input sequence itself
            alternate: Query matrix is randomly initialized (similar to target but with alternating attention heads)
            hierarchical_target: Query matrices are randomly initialized
            hierarchical_label: Query matrices are initialized using sentence embedding of code descriptions of all
                hierarchy levels
    scale : bool; default=False
        Flag indicating whether Energy Scores E (QxK.T) should be scaled using square-root(embedding_dim)
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when performing multihead attention
    num_cats : int; default=None
        Number of categories |L| in high-level label space
    label_embedding_matrix : np.array; default=None
        Sentence embedding matrix of code descriptions of the low-level label space (e.g., billable ICD-9 codes)
        E.g., 003.0: Salmonella gastroenteritis
    cat_embedding_matrix : np.array; default=None
        Sentence embedding matrix of category descriptions of the high-level category label space (e.g., ICD-9 category)
        E.g., 001-139: Infectious And Parasitic Diseases
    code2cat_map: List[int]; default=None
        List containing a index mapping of the codes (lobels) to idx of categories

    """
    def __init__(self,
                 num_labels: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 att_module: str,
                 scale: bool = False,
                 multihead: bool = False,
                 num_heads: int = None,
                 num_cats: int = None,
                 embedding_scaling: float = 1,
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None,
                 code2cat_map: List[int] = None):

        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._att_module = att_module
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self._num_cats = num_cats
        self._embedding_scaling = embedding_scaling
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map

        # Init key-value pair matrices
        # Initialize key-value pair matrices
        self.K = nn.Conv1d(in_channels=self._latent_doc_dim,
                           out_channels=self._latent_doc_dim,
                           kernel_size=1)
        nn.init.xavier_uniform_(self.K.weight)
        self.K.bias.data.fill_(0.01)

        self.V = nn.Conv1d(in_channels=self._latent_doc_dim,
                           out_channels=self._latent_doc_dim,
                           kernel_size=1)
        nn.init.xavier_uniform_(self.V.weight)
        self.V.bias.data.fill_(0.01)

        # Init multi-head attention output layer to concatenate output of all attention heads
        if self._multihead:
            self.MH_output = nn.Linear(in_features=self._latent_doc_dim,
                                       out_features=self._latent_doc_dim)

        # Label attention with different query initialization strategies
        # |L| queries create |L| label-specific latent document representations
        if self._att_module == 'random':
            self.attention_layer = RandomAttention(num_labels=self._num_labels,
                                                   embedding_dim=self._embedding_dim,
                                                   latent_doc_dim=self._latent_doc_dim,
                                                   scale=self._scale,
                                                   multihead=self._multihead,
                                                   num_heads=self._num_heads)
        elif self._att_module == 'hierarchical_random':
            self.attention_layer = HierarchicalRandomAttention(num_labels=self._num_labels,
                                                               num_cats=self._num_cats,
                                                               embedding_dim=self._embedding_dim,
                                                               latent_doc_dim=self._latent_doc_dim,
                                                               code2cat_map=self._code2cat_map,
                                                               scale=self._scale,
                                                               multihead=self._multihead,
                                                               num_heads=self._num_heads)
        elif self._att_module == 'pretrained':
            self.attention_layer = PretrainedAttention(num_labels=self._num_labels,
                                                       embedding_dim=self._embedding_dim,
                                                       latent_doc_dim=self._latent_doc_dim,
                                                       embedding_scaling=self._embedding_scaling,
                                                       label_embedding_matrix=self._label_embedding_matrix,
                                                       scale=self._scale,
                                                       multihead=self._multihead,
                                                       num_heads=self._num_heads)

        elif self._att_module == 'hierarchical_pretrained':
            self.attention_layer = HierarchicalPretrainedAttention(num_labels=self._num_labels,
                                                                   num_cats=self._num_cats,
                                                                   embedding_dim=self._embedding_dim,
                                                                   latent_doc_dim=self._latent_doc_dim,
                                                                   code2cat_map=self._code2cat_map,
                                                                   embedding_scaling=self._embedding_scaling,
                                                                   cat_embedding_matrix=self._cat_embedding_matrix,
                                                                   label_embedding_matrix=self._label_embedding_matrix,
                                                                   scale=self._scale,
                                                                   multihead=self._multihead,
                                                                   num_heads=self._num_heads)

        # target attention using a single query to create one latent document representation
        elif self._att_module == 'target':
            self.attention_layer = TargetAttention(num_labels=self._num_labels,
                                                   embedding_dim=self._embedding_dim,
                                                   latent_doc_dim=self._latent_doc_dim,
                                                   scale=self._scale,
                                                   multihead=self._multihead,
                                                   num_heads=self._num_heads)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of general attention mechanism class.

        Parameters
        ----------
        H : torch.Tensor  [batch_size, latent_doc_dim, sequence_length]
            Latent document representation after CNN, RNN, or Transformer

        Returns
        -------
        C : torch.Tensor
            Context matrix after attention mechanism
        A : torch.Tensor
            Attention weight matrix

        """
        K = F.elu(self.K(H).permute(0, 2, 1))
        V = F.elu(self.V(H).permute(0, 2, 1))
        if self._multihead:
            C, A = self.attention_layer(K=K, V=V)
            C = transpose_output(X=C, num_heads=self._num_heads)
            A = transpose_output(X=A, num_heads=self._num_heads)
            C = self.MH_output(C)
        else:
            C, A = self.attention_layer(K=K, V=V)
        return C, A
