"""
This file contains source code for hierarchical attention.
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


class HierarchicalTargetAttention(nn.Module):
    """
    Hierarchical target attention mechanism inspired by Galassi et al. (2021) - Attention in Natural Language Processing
    (source: https://arxiv.org/abs/1902.02181) representation of hierarchical attention proposed by Zhao and Zhang
    (2018) (Figure. 6 - center).

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in billable code label space
    num_cats : int
        Number of labels |L| in category label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    code2cat_map : List[int]
        List containing index mappings of codes to their respective categories
    scale : bool; default=False
        Flag indicating whether Energy Scores E (QxK.T) should be scaled using square-root(embedding_dim)
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated

    """

    def __init__(self,
                 num_labels: int,
                 num_cats: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 code2cat_map: List[int],
                 scale: bool = True,
                 multihead: bool = False,
                 num_heads: int = None):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
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

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_cats)
        nn.init.xavier_uniform_(self.Q1_mat.weight)
        self.Q1 = self.Q1_mat.weight.clone()
        #        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)

        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q2_mat.weight)
        self.Q2 = self.Q2_mat.weight.clone()

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

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:

        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

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
        Q1 = self.Q1.to(device)
        Q2 = self.Q2.to(device)

        if self._multihead:
            Q1 = torch.unsqueeze(Q1, dim=0).repeat(K.size()[0], 1, 1)
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q1 = transpose_qkv(self.W_q1(Q1), self._num_heads)
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)

            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
            A1 = F.softmax(input=E1, dim=-1)
            C1 = torch.bmm(A1, V)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                Q2[:, i, :] += C1[:, code2cat_idx, :]

            if self._scale:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)
            C2 = torch.bmm(A2, V)

        else:
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            if self._scale:
                E1 = Q1.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = Q1.matmul(K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)
            C1 = A1.matmul(V)  # output shape: [batch_size, number_categories, latent_doc_dim]
            # Map context vector of the ith category to each code belong to the same category.

            for i, code2cat_idx in enumerate(self._code2cat_map):
                Q2[:, i, :] += C1[:, code2cat_idx, :]

            if self._scale:
                E2 = Q2.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = Q2.matmul(K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)

            # Compute attention weighted document embeddings - context matrix
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
            C2 = A2.matmul(V)
        return C2, A2


class HierarchicalContextAttention(nn.Module):
    """

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in billable code label space
    num_cats : int
        Number of labels |L| in category label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    code2cat_map : List[int]
        List containing index mappings of codes to their respective categories
    scale : bool; default=False
        Flag indicating whether Energy Scores E (QxK.T) should be scaled using square-root(embedding_dim)
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated

    """

    def __init__(self,
                 num_labels: int,
                 num_cats: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 code2cat_map: List[int],
                 scale: bool = True,
                 multihead: bool = False,
                 num_heads: int = None):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
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

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_cats)
        nn.init.xavier_uniform_(self.Q1_mat.weight)
        self.Q1 = self.Q1_mat.weight.clone()
        #        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)

        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q2_mat.weight)
        self.Q2 = self.Q2_mat.weight.clone()

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

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:

        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

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
        Q1 = self.Q1.to(device)
        Q2 = self.Q2.to(device)

        if self._multihead:
            Q1 = torch.unsqueeze(Q1, dim=0).repeat(K.size()[0], 1, 1)
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q1 = transpose_qkv(self.W_q1(Q1), self._num_heads)
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)

            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
            A1 = F.softmax(input=E1, dim=-1)
            C1 = torch.bmm(A1, V)

            if self._scale:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)
            C2 = torch.bmm(A2, V)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                C2[:, i, :] += C1[:, code2cat_idx, :]

        else:
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            if self._scale:
                E1 = Q1.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = Q1.matmul(K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)
            C1 = A1.matmul(V)  # output shape: [batch_size, number_categories, latent_doc_dim]
            # Map context vector of the ith category to each code belong to the same category.

            if self._scale:
                E2 = Q2.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = Q2.matmul(K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)

            # Compute attention weighted document embeddings - context matrix
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
            C2 = A2.matmul(V)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                C2[:, i, :] += C1[:, code2cat_idx, :]
        return C2, A2


class HierarchicalDoubleAttention(nn.Module):
    """

    Parameters
    ----------
    num_labels : int
        Number of labels |L| in billable code label space
    num_cats : int
        Number of labels |L| in category label space
    embedding_dim : int
        Dimension of token embeddings
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    code2cat_map : List[int]
        List containing index mappings of codes to their respective categories
    scale : bool; default=False
        Flag indicating whether Energy Scores E (QxK.T) should be scaled using square-root(embedding_dim)
    multihead : bool; default=False
        Flag indicating if multihead attention has to be performed.
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated

    """

    def __init__(self,
                 num_labels: int,
                 num_cats: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 code2cat_map: List[int],
                 scale: bool = True,
                 multihead: bool = False,
                 num_heads: int = None):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
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

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_cats)
        nn.init.xavier_uniform_(self.Q1_mat.weight)
        self.Q1 = self.Q1_mat.weight.clone()

        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q2_mat.weight)
        self.Q2 = self.Q2_mat.weight.clone()

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

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:

        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

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
        Q1 = self.Q1.to(device)
        Q2 = self.Q2.to(device)

        if self._multihead:
            Q1 = torch.unsqueeze(Q1, dim=0).repeat(K.size()[0], 1, 1)
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q1 = transpose_qkv(self.W_q1(Q1), self._num_heads)
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)

            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
            A1 = F.softmax(input=E1, dim=-1)

            if self._scale:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                A2[:, i, :] += A1[:, code2cat_idx, :]
            A2 = A2 / 2

            C2 = torch.bmm(A2, V)

        else:
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            if self._scale:
                E1 = Q1.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = Q1.matmul(K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)

            if self._scale:
                E2 = Q2.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = Q2.matmul(K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                A2[:, i, :] += A1[:, code2cat_idx, :]
            A2 = A2 / 2

            # Compute attention weighted document embeddings - context matrix
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
            C2 = A2.matmul(V)

        return C2, A2


class HierarchicalLabelAttention(nn.Module):
    """
    Hierarchical target attention mechanism inspired by Galassi et al. (2021) - Attention in Natural Language Processing
    (source: https://arxiv.org/abs/1902.02181) representation of hierarchical attention proposed by Zhao and Zhang
    (2018) (Figure. 6 - center).

    Parameters
    ----------
    encoder_out_dim : int
        Output dimension of encoder architecture, i.e., dimension of hidden document representation
    embedding_dim : int
        Dimension of word/token embedding
    n_labels_lvl_1 : int
        Number of labels in first hierarchy level - high-level categories
    n_labels_lvl_2 : int
        Number of labels in second hierarchy level - low-level codes
    cat_embedding_matrix : np.array
        Embedding matrix pretrained on the category descriptions
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled

    """
    def __init__(self,
                 num_labels: int,
                 num_cats: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 code2cat_map: List[int],
                 cat_embedding_matrix: np.array,
                 label_embedding_matrix: np.array,
                 scale: bool = False,
                 multihead: bool = False,
                 num_heads: int = None):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
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

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_cats)
        self.Q1_mat.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)
        #        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)
        self.Q1 = self.Q1_mat.weight.clone()
        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2_mat = nn.Linear(in_features=self._latent_doc_dim,
                                out_features=self._num_labels)
        self.Q2_mat.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)
        self.Q2 = self.Q2_mat.weight.clone()

        # Conv1D layer to avoid dimensionality mismatch
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
            self.W_q1 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q1.weight)
            self.W_q1.bias.data.fill_(0.01)

            self.W_q2 = nn.Linear(in_features=self._latent_doc_dim,
                                  out_features=self._latent_doc_dim)
            nn.init.xavier_uniform_(self.W_q2.weight)
            self.W_q2.bias.data.fill_(0.01)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:

        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

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
        Q1 = self._mapping_layer(self.Q1.permute(1, 0)).permute(1, 0).to(device)
        Q2 = self._mapping_layer(self.Q2.permute(1, 0)).permute(1, 0).to(device)

        if self._multihead:
            Q1 = torch.unsqueeze(Q1, dim=0).repeat(K.size()[0], 1, 1)
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)
            K = transpose_qkv(self.W_k(K), self._num_heads)
            V = transpose_qkv(self.W_v(V), self._num_heads)
            Q1 = transpose_qkv(self.W_q1(Q1), self._num_heads)
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)
            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
            A1 = F.softmax(input=E1, dim=-1)
            C1 = torch.bmm(A1, V)

            for i, code2cat_idx in enumerate(self._code2cat_map):
                Q2[:, i, :] += C1[:, code2cat_idx, :]

            if self._scale:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)
            C2 = torch.bmm(A2, V)
        else:
            Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1).to(device)
            if self._scale:
                E1 = Q1.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E1 = Q1.matmul(K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)
            C1 = A1.matmul(V)  # output shape: [batch_size, number_categories, latent_doc_dim]
            # Map context vector of the ith category to each code belong to the same category.

            for i, code2cat_idx in enumerate(self._code2cat_map):
                Q2[:, i, :] += C1[:, code2cat_idx, :]

            if self._scale:
                E2 = Q2.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E2 = Q2.matmul(K.permute(0, 2, 1))
            A2 = F.softmax(input=E2, dim=-1)

            # Compute attention weighted document embeddings - context matrix
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
            C2 = A2.matmul(V)
        return C2, A2
