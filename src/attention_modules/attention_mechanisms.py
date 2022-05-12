"""
This file contains source code contains the pytorch implementation for the different attention mechanisms. The scripts
are written to modularize the different variations.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/11/2022

Attention mechanisms:
    - Self-attention (implemented)
    - Target-attention (implemented, tested)
    - Label-attention (implemented, tested)
    - Hierarchical-attention
    - Multi-head attention
    - Alternating attention
"""
# built-in libraries
from typing import Tuple

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAttention(nn.Module):
    """
    This class performs target attention with trainable queries or targets.

    Parameters
    ----------
    embedding_dim : int
        Number of features of input
    n_labels : int
        Number of labels considered in the label space
    scale : bool
        Flag indicating if energy scores (QxK.T) should be scaled
    """
    def __init__(self,
                 embedding_dim: int,
                 n_labels: int,
                 scale: bool = False):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._n_labels = n_labels
        self._scale = scale

        # Init target weights also considered to be the query embedding matrix Q ∈ R^nxd
        # where n: n_labels and d: output feature of latent representation of documents using CNN or LSTM
        self.U = nn.Linear(in_features=self._embedding_dim,
                           out_features=self._n_labels)
        nn.init.xavier_uniform_(self.U.weight)

    def forward(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism
        Parameters
        ----------
        K : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            Key embeddings K ∈ R^lxd
        V : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            Value embeddings V ∈ R^lxd

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K: QK.T
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        if self._scale:
            E = self.U.weight.matmul(K) / np.sqrt(self._embedding_dim)
        else:
            E = self.U.weight.matmul(K)

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A = F.softmax(input=E, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C = A.matmul(V)
        return C, A


class SelfAttention(nn.Module):
    """
    This class performs self-attention with trainable weight matrices for the Key, Query, and Value.
    
    Parameters
    ----------
    embedding_dim : int
        Number of features of input
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled
    """
    def __init__(self,
                 embedding_dim: int,
                 scale: bool = False):
        super().__init__()
        self._embedding_dim = embedding_dim  # == out_features
        self._scale = scale

        # Init 1D convolution layers to extract more meaningful features from the input sequence used in self-attention
        # Following Gao et al. 2019 (https://www.sciencedirect.com/science/article/pii/S0933365719303562)
        # K - key matrix
        self._K_conv = nn.Conv1d(in_channels=self._embedding_dim,
                                 out_channels=self._embedding_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._K_conv.weight)
        self._K_conv.bias.data.fill_(0.01)

        # Q - query matrix
        self._Q_conv = nn.Conv1d(in_channels=self._embedding_dim,
                                 out_channels=self._embedding_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._Q_conv.weight)
        self._Q_conv.bias.data.fill_(0.01)

        # V - value matrix
        self._V_conv = nn.Conv1d(in_channels=self._embedding_dim,
                                 out_channels=self._embedding_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._V_conv.weight)
        self._V_conv.bias.data.fill_(0.01)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism
        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H ∈ R^lxd

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Extract specific information from the input sequence for the key
        # H ∈ R^lxd -> K ∈ R^lxd; where l: sequence length and d: word embedding dimension
        K = F.elu(self._K_conv(H)).permute(0, 2, 1)
        # H ∈ R^lxd -> Q ∈ R^lxd; where l: sequence length and d: word embedding dimension
        Q = F.elu(self._Q_conv(H)).permute(0, 2, 1)
        # H ∈ R^lxd -> V ∈ R^lxd; where l: sequence length and d: word embedding dimension
        V = F.elu(self._V_conv(H)).permute(0, 2, 1)

        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K: QK.T
        # where e_i represents the energy score for i-th token in the input sequence
        # E ∈ R^lxl, where l: sequence length
        if self._scale:
            E = Q.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
        else:
            E = Q.matmul(K.permute(0, 2, 1))

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^lxl, where l: sequence length
        A = F.softmax(input=E, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C = A.matmul(V)
        return C, A


class LabelAttention(nn.Module):
    """
    Label Attention with pretrained label embedding matrix generated using Doc2Vec.

    Parameters
    ----------
    n_labels : int
        Number of features of input
    embedding_dim : int
        Number of features of input
    map_dim : int,
        Final dimension that the label embedding matrix is mapped to to avoid dimension mismatch
    label_embedding_matrix : np.array
        Embedding matrix pre-trained on the code descriptions associated with the samples from the training dataset
    scale: bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled

    """
    def __init__(self,
                 n_labels: int,
                 embedding_dim: int,
                 map_dim: int,
                 label_embedding_matrix: np.array,
                 scale: bool = False):
        super().__init__()
        self._n_labels = n_labels
        self._embedding_dim = embedding_dim
        self._map_dim = map_dim
        self._scale = scale

        self.Q = nn.Linear(in_features=self._embedding_dim,
                           out_features=self._n_labels,
                           bias=True)
        self.Q.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Need a 1D-Conv layer to map the embedded (with embedding dimension) code descriptions
        # to the output dimension of the parallel convolution layers
        # See (Liu et al. 2021 - EffectiveCAN - 3.3 Attention)
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._map_dim,
                                        kernel_size=1)

    def forward(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of label-attention mechanism with pretrained label embedding matrix
        Parameters
        ----------
        K : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            Key embeddings K ∈ R^lxd
        V : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            Value embeddings V ∈ R^lxd

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Map the label embedding matrix from embedding dim to dimension of convolution output
        Q = self._mapping_layer(self.Q.weight.permute(1, 0)).permute(1, 0)
        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K: QK.T
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        if self._scale:
            E = Q.matmul(K) / np.sqrt(self._embedding_dim)
        else:
            E = Q.matmul(K)

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A = F.softmax(input=E, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C = A.matmul(V)
        return C, A

