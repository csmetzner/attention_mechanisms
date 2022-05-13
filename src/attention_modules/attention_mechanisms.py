"""
This file contains source code contains the pytorch implementation for the different attention mechanisms. The scripts
are written to modularize the different variations.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/13/2022

Attention mechanisms:
    - Self-attention (implemented, tested)
    - Target-attention (implemented, tested)
    - Label-attention (implemented, tested)
    - Hierarchical-attention
        - Target attention (implemented, tested)
        - Label attention (implemented, tested)
    - Multi-head attention
    - Alternating attention (implemented, tested)
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
    Target attention with trainable query matrices.

    Parameters
    ----------
    encoder_out_dim : int
        Output dimension of encoder architecture, i.e., dimension of hidden document representation
    n_att_vectors : int
        Number of attention vectors, e.g., number of labels in the label space |L|
    embedding_dim : int
        Embedding dimension of tokens
    scale : bool
        Flag indicating if energy scores (QxK.T) should be scaled by the root of

    """
    def __init__(self,
                 encoder_out_dim: int,
                 n_att_vectors: int,
                 embedding_dim: int,
                 scale: bool = False):

        super().__init__()
        self._encoder_out_dim = encoder_out_dim
        self._n_att_vectors = n_att_vectors
        self._embedding_dim = embedding_dim
        self._scale = scale

        # Initialize target matrix (i.e., query embedding matrix Q ∈ R^nxd)
        # where n: number of attention vectors (e.g., number of labels)
        # and d: output dimension of latent document representation of CNN/LSTM/Transformer
        self.Q = nn.Linear(in_features=self._encoder_out_dim,
                           out_features=self._n_att_vectors)
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H is used as: | Key embedding matrix - K ∈ R^lxd | Value embedding matrix - V ∈ R^lxd |

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K(H): QK.T
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        if self._scale:
            E = self.Q.weight.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E = self.Q.weight.matmul(H)

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A = F.softmax(input=E, dim=2)

        # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent document representation dimension of CNN/LSTM/Transformer
        C = A.matmul(H.permute(0, 2, 1))

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
                 encoder_out_dim: int,
                 embedding_dim: int,
                 scale: bool = False):
        super().__init__()
        self._encoder_out_dim = encoder_out_dim
        self._embedding_dim = embedding_dim
        self._scale = scale

        # Init 1D convolution layers to extract more meaningful features from the input sequence used in self-attention
        # Following Gao et al. 2019 (https://www.sciencedirect.com/science/article/pii/S0933365719303562)
        # K - key matrix
        self._K_conv = nn.Conv1d(in_channels=self._encoder_out_dim,
                                 out_channels=self._encoder_out_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._K_conv.weight)
        self._K_conv.bias.data.fill_(0.01)

        # Q - query matrix
        self._Q_conv = nn.Conv1d(in_channels=self._encoder_out_dim,
                                 out_channels=self._encoder_out_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._Q_conv.weight)
        self._Q_conv.bias.data.fill_(0.01)

        # V - value matrix
        self._V_conv = nn.Conv1d(in_channels=self._encoder_out_dim,
                                 out_channels=self._encoder_out_dim,
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

        # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^lxd, where l: sequence length and d: latent dimension of CNN/LSTM model
        C = A.matmul(V)
        return C, A


class LabelAttention(nn.Module):
    """
    Label Attention with pretrained label embedding matrix generated using Doc2Vec.

    Parameters
    ----------
    encoder_out_dim : int
        Output dimension of encoder architecture, i.e., dimension of hidden document representation
    n_labels : int
        Number of attention vectors, e.g., number of labels in the label space |L|
    embedding_dim : int
        Embedding dimension of tokens
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    scale: bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled

    """

    def __init__(self,
                 encoder_out_dim: int,
                 embedding_dim: int,
                 n_labels: int,
                 label_embedding_matrix: np.array,
                 scale: bool = False):
        super().__init__()
        self._encoder_out_dim = encoder_out_dim
        self._embedding_dim = embedding_dim
        self._n_labels = n_labels
        self._scale = scale

        # Init label embedding matrix by using linear layer
        # Q ∈ R^nxd_e where n: number of labels in |L| and d_e: embedding dimension of tokens
        self.Q = nn.Linear(in_features=self._embedding_dim,
                           out_features=self._n_labels,
                           bias=True)
        self.Q.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Need a 1D-Conv layer to map the embedded (with embedding dimension) code descriptions
        # to the output dimension of the parallel convolution layers
        # See (Liu et al. 2021 - EffectiveCAN - 3.3 Attention)
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._encoder_out_dim,
                                        kernel_size=1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of label-attention mechanism with pretrained label embedding matrix.

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H is used as: | Key embedding matrix - K ∈ R^lxd | Value embedding matrix - V ∈ R^lxd |

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
        # Compute energy score matrix E - dot product of query embeddings Q and key embeddings K(H): QK.T
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        if self._scale:
            E = Q.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E = Q.matmul(H)

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A = F.softmax(input=E, dim=2)

        # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C = A.matmul(H.permute(0, 2, 1))
        return C, A


class AlternateAttention(nn.Module):
    """
    Alternate attention mechanism as proposed by Bi et al. 2020 - Imbalanced Chinese Multi-label Text Classification
    Based on Alternating Attention (https://aclanthology.org/2020.paclic-1.42/)

    Parameters
    ----------
    encoder_out_dim : int
        Number of features of input
    n_att_vectors : int
        Number of features of output - number of attention hops (1 hop = 1 attention score vector)
        could be number of labels in label space |L|
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled

    """

    def __init__(self,
                 encoder_out_dim,
                 n_att_vectors,
                 scale):
        super().__init__()
        self._encoder_out_dim = encoder_out_dim
        self._n_att_vectors = n_att_vectors
        self._scale = scale

        # Init target weights also considered to be the query embedding matrix Q ∈ R^nxd
        # where n: n_labels and d: output feature of latent representation of documents using CNN or LSTM
        # First alternating attention head - M
        self.M = nn.Linear(in_features=self._encoder_out_dim,
                           out_features=self._n_att_vectors)
        nn.init.xavier_uniform_(self.M.weight)

        # Second alternating attention head - N
        self.N = nn.Linear(in_features=self._encoder_out_dim,
                           out_features=self._n_att_vectors)
        nn.init.xavier_uniform_(self.N.weight)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of alternating attention heads

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H is used as: | Key embedding matrix - K ∈ R^lxd | Value embedding matrix - V ∈ R^lxd |

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
        if self._scale:
            EM = self.M.weight.matmul(H) / np.sqrt(self._embedding_dim)
            EN = self.N.weight.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            EM = self.M.weight.matmul(H)
            EN = self.N.weight.matmul(H)

        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        AM = F.softmax(input=EM, dim=2)
        AN = F.softmax(input=EN, dim=2)

        # Create mask to set alternating elements in AM and AN to 0
        mask_m = torch.randn(AM.size()).bool()
        for i in range(AM.size()[-1]):
            if i % 2 != 0:
                mask_m[:, :, i] = False
        AM = AM.mul(mask_m)  # set every second element to 0 starting at index 1

        mask_n = torch.randn(AN.size()).bool()
        for i in range(AN.size()[-1]):
            if i % 2 == 0:
                mask_n[:, :, i] = False
        AN = AN.mul(mask_n)  # set every second element to 0 starting at index 0

        # Combine AM and AN
        A = F.relu(AM + AN)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C = A.matmul(H.permute(0, 2, 1))
        return C, A


class HierarchicalTargetAttention(nn.Module):
    """
    Hierarchical target attention mechanism inspired by Galassi et al. (2021) - Attention in Natural Language Processing
    (source: https://arxiv.org/abs/1902.02181) representation of hierarchical attention proposed by Zhao and Zhang
    (2018) (Figure. 6 - center).

    Parameters
    ----------
    encoder_out_dim : int
        Output dimension of encoder architecture, i.e., dimension of hidden document representation
    n_labels_lvl_1 : int
        Number of labels in first hierarchy level - high-level categories
    n_labels_lvl_2 : int
        Number of labels in second hierarchy level - low-level codes
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled

    """

    def __init__(self,
                 encoder_out_dim: int,
                 n_labels_lvl_1: int,
                 n_labels_lvl_2: int,
                 scale: bool = False):
        super().__init__()
        self._encoder_out_dim = encoder_out_dim
        self._n_labels_lvl_1 = n_labels_lvl_1
        self._n_labels_lvl_2 = n_labels_lvl_2
        self._scale = scale

        # Init query (target) matrices for hierarchical attention using weights of linear layers
        # Level 1: high-level label category attention
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: output dimension of encoder
        self.Q1 = nn.Linear(in_features=self._encoder_out_dim,
                            out_features=self._n_labels_lvl_1)
        nn.init.xavier_uniform_(self.Q1.weight)

        # Level 2: low-level label category attention
        # Q2 ∈ R^n2xd where n2: number of labels of low-level codes and d: output dimension of encoder
        self.Q2 = nn.Linear(in_features=self._encoder_out_dim,
                            out_features=self._n_labels_lvl_2)
        nn.init.xavier_uniform_(self.Q2.weight)

        # Init third linear layer to map dimension (number of labels of level 1) context vector of level 1 (C1) to
        # have the same dimension of query matrix Q2
        # Q2 ∈ R^n2xn1 where n2: number of labels of low-level codes and n1: number of labels of high-level categories
        self.Q12 = nn.Linear(in_features=self._n_labels_lvl_1,
                             out_features=self._n_labels_lvl_2)
        nn.init.xavier_uniform_(self.Q12.weight)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of hierarchical target attention mechanism.

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H is used as: | Key embedding matrix - K ∈ R^lxd | Value embedding matrix - V ∈ R^lxd |

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """

        # Forward pass of Level 1:
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        if self._scale:
            E1 = self.Q1.weight.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E1 = self.Q1.weight.matmul(H)
        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A1 = F.softmax(input=E1, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C1 = A1.matmul(H.permute(0, 2, 1))

        # Use generated context vectors to inform query matrix of second of low-level hierarchy
        C1 = self.Q12.weight.matmul(C1)
        Q2 = self.Q2.weight.mul(C1)

        if self._scale:
            E2 = Q2.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E2 = Q2.matmul(H)

        A2 = F.softmax(input=E2, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C2 = A2.matmul(H.permute(0, 2, 1))
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
                 encoder_out_dim: int,
                 embedding_dim: int,
                 n_labels_lvl_1: int,
                 n_labels_lvl_2: int,
                 cat_embedding_matrix: np.array,
                 label_embedding_matrix: np.array,
                 scale: bool = False):
        super().__init__()
        self._encoder_out_dim = encoder_out_dim  # E.g., Output dimension of CNN, RNN, Transformer
        self._embedding_dim = embedding_dim
        self._n_labels_lvl_1 = n_labels_lvl_1  # E.g., number of labels of high-level categories
        self._n_labels_lvl_2 = n_labels_lvl_2  # E.g., number of labels of low-level codes
        self._scale = scale

        # Init hierarchical attention layers
        # Level 1: high-level label category attention
        self.Q1 = nn.Linear(in_features=self._encoder_out_dim,
                            out_features=self._n_labels_lvl_1)
        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)

        # Level 1-2: Map context vector to number of labels of level 2
        # treated as intermediate layer
        self.Q12 = nn.Linear(in_features=self._n_labels_lvl_1,
                             out_features=self._n_labels_lvl_2)
        nn.init.xavier_uniform_(self.Q12.weight)

        # Level 2: low-level label category attention
        self.Q2 = nn.Linear(in_features=self._encoder_out_dim,
                            out_features=self._n_labels_lvl_2)
        self.Q2.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Conv1D layer to avoid dimensionality mismatch
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._encoder_out_dim,
                                        kernel_size=1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of hierarchical target attention mechanism.

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]
            H is used as: | Key embedding matrix - K ∈ R^lxd | Value embedding matrix - V ∈ R^lxd |

        Returns
        -------
        C : torch.Tensor
            Context matrix C - adjusted document embeddings
            where c_i represents the context vector for the i-th label in the label space
        A : torch.Tensor
            Attention weight matrix A containing the attention scores
            where a_i represents the attention weight for the i-th label in the label space

        """
        # Level 1:
        # Map the label embedding matrix from embedding dim to dimension of convolution output
        Q1 = self._mapping_layer(self.Q1.weight.permute(1, 0)).permute(1, 0)
        print(f'Q1: {Q1.size()}')
        # where e_i represents the energy score for i-th label in the label space
        # E ∈ R^nxl, where n: number of labels and l: sequence length
        print(f'H: {H.size()}')
        if self._scale:
            E1 = Q1.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E1 = Q1.matmul(H)
        # Compute attention weights matrix A using a distribution function g (here softmax)
        # where a_i represents the attention weights for the i-th label in the label space
        # A ∈ R^nxl, where n: number of labels and l: sequence length
        A1 = F.softmax(input=E1, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C1 = A1.matmul(H.permute(0, 2, 1))
        # Use context vectors to inform Query matrix of second of low-level hierarchy
        # Map context vector C1 to have |L2| rows
        C1 = self.Q12.weight.matmul(C1)

        # Need to map dimension of description embeddings to fit output dimension of latent doc representation
        Q2 = self._mapping_layer(self.Q2.weight.permute(1, 0)).permute(1, 0)
        Q2 = Q2.mul(C1)
        if self._scale:
            E2 = Q2.matmul(H) / np.sqrt(self._embedding_dim)
        else:
            E2 = Q2.matmul(H)

        A2 = F.softmax(input=E2, dim=2)

        # Compute attention weighted document embeddings - context matrix
        # Where c_i represents the document context vector for the i-th label in the label space
        # C ∈ R^nxd, where n: number of labels and d: latent dimension of CNN/LSTM model
        C2 = A2.matmul(H.permute(0, 2, 1))
        return C2, A2