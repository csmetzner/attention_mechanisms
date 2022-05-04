"""
This file contains source code contains the pytorch implementation for the different attention mechanisms. The scripts
are written to modularize the different variations.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/03/2022

Attention mechanisms:
    - Self-attention
    - Target-attention (implemented)
    - Label-attention
    - Hierarchical-attention
    - Multi-head attention
    - Alternating attention
"""

# installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAttention(nn.Module):
    """
    This class performs target attention with trainable queries or targets.

    Parameters
    ----------
    in_features : int
        Number of features of input
    n_labels : int
        Number of labels considered in the label space
    """
    def __init__(self,
                 in_features: int,
                 n_labels: int):
        super().__init__()
        self._in_features = in_features
        self._n_labels = n_labels

        # Init target weights also considered to be the query embedding matrix Q ∈ R^nxd
        # where n: n_labels and d: output feature of latent representation of documents using CNN or LSTM
        self.U = nn.Linear(in_features=self._in_features,
                           out_features=self._n_labels)
        nn.init.xavier_uniform_(self.U.weight)

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of target attention mechanism
        Parameters
        ----------
        K : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, embedding_dim: d]
            Key embeddings K ∈ R^lxd

        Returns
        -------
        torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for i-th label in the label space

        """
        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K: QK.T
        # where e_i represents the energy score for i-th label in the label space
        E = torch.matmul(input=self.U.weight, other=K.permute(0, 2, 1))  # E ∈ R^nxl, where n: n_labels and l: sequence length

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        A = F.softmax(input=E, dim=2)

        return A





