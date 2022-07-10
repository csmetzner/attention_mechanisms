"""
This file contains source code for a simple implementation of alternating attention.

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


class AlternateAttention(nn.Module):
    """
    Alternate attention mechanism as proposed by Bi et al. 2020 - Imbalanced Chinese Multi-label Text Classification
    Based on Alternating Attention (https://aclanthology.org/2020.paclic-1.42/)

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

        # Init target weights also considered to be the query embedding matrix Q ∈ R^nxd
        # where n: n_labels and d: output feature of latent representation of documents using CNN or LSTM
        # First alternating attention head - M
        self.Q1 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q1.weight)

        # Second alternating attention head - N
        self.Q2 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q2.weight)

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
            self.W_q1 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q1.weight)
            self.W_q1.bias.data.fill_(0.01)

            self.W_q2 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q2.weight)
            self.W_q2.bias.data.fill_(0.01)

    def forward(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism

        Parameters
        ----------
        H : torch.Tensor
            Latent document representation - H ∈ R^lxd; where l: sequence length and d: latent document dimension

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Compute energy scores for both alternating attention heads
        # Head 1: EM - dot product of query embedding matrix Q and key embedding matrix K: QxK.T
        # where em_i represents the energy score for i-th attention vector; every second value is set to 0
        # Head 2: EN - dot product of query embedding matrix Q and key embedding matrix K: QxK.T
        # where en_i represents the energy score for i-th attention vector; every other second value is set to 0
        # E ∈ R^nxl, where n: dimension of attention vector and l: sequence length
        if self._multihead:
            Q1 = torch.unsqueeze(self.Q1.weight, dim=0).repeat(K.size()[0], 1, 1)
            Q2 = torch.unsqueeze(self.Q2.weight, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q1 = transpose_qkv(self.W_q1(Q1), self._num_heads)
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)

            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)

            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)
            A2 = F.softmax(input=E2, dim=-1)

            # Create mask to set alternating elements in AM and AN to 0
            mask_1 = torch.randn(A1.size()).bool().to(device)
            for i in range(A1.size()[-1]):
                if i % 2 != 0:
                    mask_1[:, :, i] = False
            A1 = A1.mul(mask_1)  # set every second element to 0 starting at index 1

            mask_2 = torch.randn(A2.size()).bool().to(device)
            for i in range(A2.size()[-1]):
                if i % 2 == 0:
                    mask_2[:, :, i] = False
            A2 = A2.mul(mask_2)  # set every second element to 0 starting at index 0

            # Combine AM and AN
            A = F.relu(A1 + A2)
            C = torch.bmm(A, V)
        else:
            Q1 = self.Q1.weight
            Q2 = self.Q2.weight
            if self._scale:
                E1 = Q1.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
                E2 = Q2.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = Q1.matmul(K.permute(0, 2, 1))
                E2 = Q2.matmul(K.permute(0, 2, 1))

            # Compute attention weights matrix A using a distribution function g (here softmax)
            # where a_i represents the attention weights for the i-th label in the label space
            # A ∈ R^nxl, where n: number of labels and l: sequence length
            A1 = F.softmax(input=E1, dim=2)
            A2 = F.softmax(input=E2, dim=2)

            # Create mask to set alternating elements in AM and AN to 0
            mask_1 = torch.randn(A1.size()).bool().to(device)
            for i in range(A1.size()[-1]):
                if i % 2 != 0:
                    mask_1[:, :, i] = False
            A1 = A1.mul(mask_1)  # set every second element to 0 starting at index 1

            mask_2 = torch.randn(A2.size()).bool().to(device)
            for i in range(A2.size()[-1]):
                if i % 2 == 0:
                    mask_2[:, :, i] = False
            A2 = A2.mul(mask_2)  # set every second element to 0 starting at index 0

            # Combine AM and AN
            A = F.relu(A1 + A2)

            # Compute attention weighted document embeddings - context matrix
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
            C = A.matmul(V)
        return C, A