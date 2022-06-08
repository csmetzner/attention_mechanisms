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
from attention_modules.multihead_attention import transpose_qkv, transpose_output

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
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None,
                 code2cat_map: List[int] = None,
                 gamma: float = None):

        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._att_module = att_module
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self._num_cats = num_cats
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map
        self._gamma = gamma

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

        if self._att_module == 'target':
            self.attention_layer = TargetAttention(num_labels=self._num_labels,
                                                   embedding_dim=self._embedding_dim,
                                                   latent_doc_dim=self._latent_doc_dim,
                                                   scale=self._scale,
                                                   multihead=self._multihead,
                                                   num_heads=self._num_heads)
            self.Q = self.attention_layer.Q.weight.clone()

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
        K = self.K(H).permute(0, 2, 1)
        V = self.V(H).permute(0, 2, 1)

        if self._att_module == 'target':
            Q = self.Q
            Q = Q.to(device)

        if self._multihead:
            C, A = self.attention_layer(H=H)
            C = transpose_output(X=C, num_heads=self._num_heads)
            A = transpose_output(X=A, num_heads=self._num_heads)
            C = self.MH_output(C)
        else:
            C, A = self.attention_layer(K=K, V=V, Q=Q)
        return C, A


class TargetAttention(nn.Module):
    """
    Target attention with trainable query matrices.

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    scale : bool; default=False
        Flag indicating whether Energy Scores E (QxK.T) should be scaled using square-root(embedding_dim)
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated
    """
    def __init__(self,
                 num_labels: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 scale: bool = False,
                 multihead: bool = False,
                 num_heads: int = None):

        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self.Q_alignment = None

        # Initialze query embedding matrix
        self.Q = nn.Linear(in_features=self._latent_doc_dim,
                           out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q.weight)

        # If multihead-attention then init additional weight layers
        if self._multihead:
            # Init key-value embedding matrix pairs
            self.W_k = nn.Linear(in_features=self._latent_doc_dim,
                                 out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_k.weight)
            self.W_k.bias.data.fill_(0.01)

            self.W_v = nn.Linear(in_features=self._latent_doc_dim,
                                 out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_v.weight)
            self.W_v.bias.data.fill_(0.01)

            # Init query embedding matrix
            self.W_q = nn.Linear(in_features=self._latent_doc_dim,
                                 out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q.weight)
            self.W_q.bias.data.fill_(0.01)

    def forward(self, K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism

        Parameters
        ----------
        K : torch.Tensor
            Key matrix with shape [batch_size, embedding_dim, sequence_length]
        V : torch.Tensor
            Value matrix with shape [batch_size, embedding_dim, sequence_length]
        Q : torch.Tensor
            Query matrix with shape [batch_size, embedding_dim, number_queries]

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        if self._multihead:
            Q = torch.unsqueeze(Q, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q = transpose_qkv(self.W_q(Q), self._num_heads)
            if self._scale:
                E = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(Q, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)
        else:
            # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K(H): QK.T
            # where e_i represents the energy score for i-th label in the label space
            # E ∈ R^nxl where n: number of labels and l: sequence length
            if self._scale:
                E = Q.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = Q.matmul(K.permute(0, 2, 1))

            # Compute attention weights matrix A using a distribution function g (here softmax)
            # where a_i represents the attention weights for the i-th label in the label space
            # A ∈ R^nxl, where n: number of labels and l: sequence length
            A = F.softmax(input=E, dim=-1)

            # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent document dimension
            C = A.matmul(V)

        return C, A