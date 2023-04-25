"""
This file contains source code for a simple implementation of label attention. Here label-attention
utilizes a query-matrix pre-trained on the code descriptions.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/31/2022
    @last modified: 05/31/2022

"""

# built-in libraries
from typing import Tuple

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.multihead_attention import transpose_qkv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PretrainedAttention(nn.Module):
    """
    Label Attention with pretrained label embedding matrix generated using Doc2Vec.

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    scale: bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated
    """

    def __init__(self,
                 num_labels: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 embedding_scaling: float,
                 label_embedding_matrix: np.array,
                 scale: bool = True,
                 multihead: bool = False,
                 num_heads: int = None):
        super().__init__()
        print('Attention mechanism: pretrained label attention')

        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._embedding_scaling = embedding_scaling
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads

        # Init label embedding matrix by using linear layer
        # Q ∈ R^nxd_e where n: number of labels in |L| and d_e: embedding dimension of tokens
        label_embedding_matrix -= label_embedding_matrix.mean()
        label_embedding_matrix /= (label_embedding_matrix.std() * self._embedding_scaling)
        self.Q = nn.Linear(in_features=self._embedding_dim,
                           out_features=self._num_labels)
        self.Q.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Need a 1D-Conv layer to map the embedded (with embedding dimension) code descriptions
        # to the output dimension of the parallel convolution layers
        # See (Liu et al. 2021 - EffectiveCAN - 3.3 Attention)
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._latent_doc_dim,
                                        kernel_size=1)

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

    def forward(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism

        Parameters
        ----------
        K : torch.Tensor
            Key matrix with shape [batch_size, embedding_dim, sequence_length]
        V : torch.Tensor
            Value matrix with shape [batch_size, embedding_dim, sequence_length]

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
            Q = torch.unsqueeze(self.Q.weight, dim=0).repeat(K.size()[0], 1, 1)
            Q = self._mapping_layer(Q.permute(0, 2, 1)).permute(0, 2, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q = transpose_qkv(self.W_q(Q), self._num_heads)
            if self._scale:
                E = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(self._latent_doc_dim)
            else:
                E = torch.bmm(Q, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)
            return C
        else:
            Q = self._mapping_layer(self.Q.weight.permute(1, 0)).permute(1, 0)
            if self._scale:
                E = Q.matmul(K.permute(0, 2, 1)) / np.sqrt(self._latent_doc_dim)
            else:
                E = Q.matmul(K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = A.matmul(V)

            return C, E


class WeightedLabelAttention(nn.Module):
    """
    Class for label-wise attention mechanism.

    Parameters
    ----------
    num_classes : int
        Number of classes in label space for given dataset
    embedding_dim : int
        Dimension of word embeddings
    hidden_dim : int
        Dimension of hidden dimension of text-encoder architecture output

    """

    def __init__(self, num_classes: int, embedding_dim: int, hidden_dim: int, label_embedding_matrix: np.array):
        super().__init__()
        self._num_classes = num_classes
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim

        # Initialize key (K) and value (V) matrices
        self.K = nn.Conv1d(in_channels=self._hidden_dim,
                           out_channels=self._hidden_dim,
                           kernel_size=1)
        nn.init.xavier_uniform_(self.K.weight)
        self.K.bias.data.fill_(0.01)

        self.V = nn.Conv1d(in_channels=self._hidden_dim,
                           out_channels=self._hidden_dim,
                           kernel_size=1)
        nn.init.xavier_uniform_(self.V.weight)
        self.V.bias.data.fill_(0.01)

        # Initialize query (Q) matrix - there is one query for the entire label space
        self.Q_rdm = nn.Linear(in_features=self._hidden_dim,
                               out_features=self._num_classes)
        nn.init.xavier_uniform_(self.Q_rdm.weight)

        self.Q_pt = nn.Linear(in_features=self._hidden_dim,
                              out_features=self._num_classes)
        self.Q_pt.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._hidden_dim,
                                        kernel_size=1)

        # Initialize weights controlling the contribution of random or pretrained embeddings
        self._alphas = nn.Parameter(data=torch.rand(self._num_classes, 1))

    def forward(self, K, V) -> torch.tensor:
        """
        Forward pass of Target Attention Mechanism Class

        Parameters
        ----------
        H : torch.tensor
            Hidden representation matrix after text-encoder architecture

        Returns
        -------
        torch.tensor
            Context vector matrix for a given batch of documents

        """
        Q_pt = self._mapping_layer(self.Q_pt.weight.permute(1, 0)).permute(1, 0)

        Q = self._alphas * self.Q_rdm.weight + (1 - self._alphas) * Q_pt

        # Scaled matrix product to compute raw energy scores
        E = Q.matmul(K.permute(0, 2, 1)) / np.sqrt(self._hidden_dim)

        # Computation of attention scores using softmax function
        A = F.softmax(input=E, dim=2)

        # Computation of the context vector representation for each document in the batch
        C = A.matmul(V)

        return C