"""
This file contains source code for context attention.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/31/2022
    @last modified: 05/31/2022
"""

# built-in libraries
from typing import List, Tuple

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libraries
from attention_modules.multihead_attention import transpose_qkv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class ContextAttention(nn.Module):
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

        # Initialize query embedding matrix
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

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
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
        K = F.elu(self.K(H)).permute(0, 2, 1)
        V = F.elu(self.V(H)).permute(0, 2, 1)

        if self._multihead:
            Q = torch.unsqueeze(self.Q.weight, dim=0).repeat(K.size()[0], 1, 1).to(device)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q = transpose_qkv(self.W_q(Q), self._num_heads)
            if self._scale:
                E = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(Q, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)

            if self._scale:
                E = torch.bmm(C, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(C, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)
        else:
            # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K(H): QK.T
            # where e_i represents the energy score for i-th label in the label space
            # E ∈ R^nxl where n: number of labels and l: sequence length
            if self._scale:
                E = self.Q.weight.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = self.Q.weight.matmul(K.permute(0, 2, 1))
            # Compute attention weights matrix A using a distribution function g (here softmax)
            # where a_i represents the attention weights for the i-th label in the label space
            # A ∈ R^nxl, where n: number of labels and l: sequence length
            A = F.softmax(input=E, dim=2)
            # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent document dimension
            C = A.matmul(V)

            # Use learned context vectors as queries
            if self._scale:
                E = C.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = C.matmul(K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=2)
            C = A.matmul(V)

        return C, A


class ContextAttentionDiffInput(nn.Module):
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

        # Initialize key-value pair matrices
        self.K1 = nn.Conv1d(in_channels=self._latent_doc_dim,
                            out_channels=self._latent_doc_dim,
                            kernel_size=1)
        nn.init.xavier_uniform_(self.K1.weight)
        self.K1.bias.data.fill_(0.01)

        self.V1 = nn.Conv1d(in_channels=self._latent_doc_dim,
                            out_channels=self._latent_doc_dim,
                            kernel_size=1)
        nn.init.xavier_uniform_(self.V1.weight)
        self.V1.bias.data.fill_(0.01)

        self.K2 = nn.Conv1d(in_channels=self._latent_doc_dim,
                            out_channels=self._latent_doc_dim,
                            kernel_size=1)
        nn.init.xavier_uniform_(self.K2.weight)
        self.K2.bias.data.fill_(0.01)

        self.V2 = nn.Conv1d(in_channels=self._latent_doc_dim,
                            out_channels=self._latent_doc_dim,
                            kernel_size=1)
        nn.init.xavier_uniform_(self.V2.weight)
        self.V2.bias.data.fill_(0.01)

        # Initialze query embedding matrix
        self.Q = nn.Linear(in_features=self._latent_doc_dim,
                           out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q.weight)

        # If multihead-attention then init additional weight layers
        if self._multihead:
            # Init key-value embedding matrix pairs
            self.W_k1 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_k1.weight)
            self.W_k1.bias.data.fill_(0.01)

            self.W_v1 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_v1.weight)
            self.W_v1.bias.data.fill_(0.01)

            self.W_k2 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_k2.weight)
            self.W_k2.bias.data.fill_(0.01)

            self.W_v2 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_v2.weight)
            self.W_v2.bias.data.fill_(0.01)

            # Init query embedding matrix
            self.W_q = nn.Linear(in_features=self._latent_doc_dim,
                                 out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q.weight)
            self.W_q.bias.data.fill_(0.01)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
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
        K1 = F.elu(self.K1(H)).permute(0, 2, 1)
        V1 = F.elu(self.V1(H)).permute(0, 2, 1)
        K2 = F.elu(self.K2(H)).permute(0, 2, 1)
        V2 = F.elu(self.V2(H)).permute(0, 2, 1)

        if self._multihead:
            Q = torch.unsqueeze(self.Q.weight, dim=0).repeat(K1.size()[0], 1, 1).to(device)
            K1 = transpose_qkv(self.W_k1(K1), self._num_heads)
            V1 = transpose_qkv(self.W_v1(V1), self._num_heads)
            K2 = transpose_qkv(self.W_k2(K2), self._num_heads)
            V2 = transpose_qkv(self.W_v2(V2), self._num_heads)
            Q = transpose_qkv(self.W_q(Q), self._num_heads)

            if self._scale:
                E = torch.bmm(Q, K1.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(Q, K1.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V1)

            if self._scale:
                E = torch.bmm(C, K2.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(C, K2.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V2)

        else:
            # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K(H): QK.T
            # where e_i represents the energy score for i-th label in the label space
            # E ∈ R^nxl where n: number of labels and l: sequence length
            if self._scale:
                E = self.Q.weight.matmul(K1.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = self.Q.weight.matmul(K1.permute(0, 2, 1))
            # Compute attention weights matrix A using a distribution function g (here softmax)
            # where a_i represents the attention weights for the i-th label in the label space
            # A ∈ R^nxl, where n: number of labels and l: sequence length
            A = F.softmax(input=E, dim=2)
            # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent document dimension
            C = A.matmul(V1)

            # Use learned context vectors as queries
            if self._scale:
                E = C.matmul(K2.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = C.matmul(K2.permute(0, 2, 1))
            A = F.softmax(input=E, dim=2)
            C = A.matmul(V2)

        return C, A

