"""
This file contains source code contains the pytorch implementation for the different attention mechanisms. The scripts
are written to modularize the different variations.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/19/2022

Attention mechanisms:
    - Self-attention (implemented, tested)
    - Target-attention (implemented, tested)
    - Label-attention (implemented, tested)
    - Hierarchical-attention
        - Target attention (implemented, tested)
        - Label attention (implemented, tested)
    - Multi-head attention (implemented, tested)
    - Alternating attention (implemented, tested)
"""
# built-in libraries
from typing import Tuple, Union, List

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map

        # Check multihead variables
        if self._multihead:
            if self._num_heads is None:
                raise ValueError('If multihead is "True" enter an integer as the number of attention heads!')
            else:
                if self._num_heads < 2:
                    raise ValueError('If multihead is "True" than num_heads must be greater then 1!')
                if self._latent_doc_dim % self._num_heads != 0:
                    raise ValueError('Select another number of attention heads!'
                                     '\nLatent_doc_dim // num_heads == 0!')



        # Initialize 1D convolution layer to map latent document representation to either the key embedding matrix (K)
        # or the value embedding matrix (V)
        self.K_1Dlayer = AttentionInputs(latent_doc_dim=self._latent_doc_dim)
        self.V_1Dlayer = AttentionInputs(latent_doc_dim=self._latent_doc_dim)

        # Initialize multihead class
        # query_dim is different for selfattention compared to the other attention mechanisms
        if self._multihead:
            self.multihead_weights = MultiHeadAttention(num_heads=self._num_heads,
                                                        key_dim=self._latent_doc_dim,
                                                        query_dim=self._latent_doc_dim if self._att_module == 'self' else self._num_labels,
                                                        value_dim=self._latent_doc_dim,
                                                        hidden_dim=self._latent_doc_dim,
                                                        att_module=self._att_module,
                                                        query_dim_cats=self._num_cats)

        if self._att_module == 'target':
            self.attention_layer = TargetAttention(num_labels=self._num_labels,
                                                   embedding_dim=self._embedding_dim,
                                                   latent_doc_dim=self._latent_doc_dim,
                                                   scale=self._scale,
                                                   multihead=self._multihead)
            self.Q = self.attention_layer.Q.weight.clone()  # retrieve query embedding matrix (Q)
        elif self._att_module == 'label':
            self.attention_layer = LabelAttention(num_labels=self._num_labels,
                                                  embedding_dim=self._embedding_dim,
                                                  latent_doc_dim=self._latent_doc_dim,
                                                  label_embedding_matrix=self._label_embedding_matrix,
                                                  scale=self._scale,
                                                  multihead=self._multihead)
            self.Q = self.attention_layer.Q.weight.clone()  # retrieve query embedding matrix (Q)
        elif self._att_module == 'self':
            self.attention_layer = SelfAttention(num_labels=self._num_labels,
                                                 embedding_dim=self._embedding_dim,
                                                 latent_doc_dim=self._latent_doc_dim,
                                                 scale=self._scale,
                                                 multihead=self._multihead)
            self.Q_1Dlayer = AttentionInputs(latent_doc_dim=self._latent_doc_dim)  # init layer to update Q
        elif self._att_module == 'alternate':
            self.attention_layer = AlternateAttention(num_labels=self._num_labels,
                                                      embedding_dim=self._embedding_dim,
                                                      latent_doc_dim=self._latent_doc_dim,
                                                      scale=self._scale,
                                                      multihead=self._multihead)
            self.Q1 = self.attention_layer.Q1.weight.clone()  # init
            self.Q2 = self.attention_layer.Q2.weight.clone()
        elif self._att_module == 'hierarchical_target':
            self.attention_layer = HierarchicalTargetAttention(num_labels=self._num_labels,
                                                               num_cats=self._num_cats,
                                                               embedding_dim=self._embedding_dim,
                                                               latent_doc_dim=self._latent_doc_dim,
                                                               code2cat_map=self._code2cat_map,
                                                               scale=self._scale,
                                                               multihead=self._multihead)
            self.Q1 = self.attention_layer.Q1.weight.clone()
            self.Q2 = self.attention_layer.Q2.weight.clone()
        elif self._att_module == 'hierarchical_label':
            self.attention_layer = HierarchicalLabelAttention(num_labels=self._num_labels,
                                                              num_cats=self._num_cats,
                                                              embedding_dim=self._embedding_dim,
                                                              latent_doc_dim=self._latent_doc_dim,
                                                              code2cat_map=self._code2cat_map,
                                                              cat_embedding_matrix= self._cat_embedding_matrix,
                                                              label_embedding_matrix= self._label_embedding_matrix,
                                                              scale=self._scale,
                                                              multihead=self._multihead)
            self.Q1 = self.attention_layer.Q1.weight.clone()
            self.Q2 = self.attention_layer.Q2.weight.clone()

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
        # 1a. Compute key (K) and value (V) matrices
        K = self.K_1Dlayer(H).permute(0, 2, 1)
        V = self.V_1Dlayer(H).permute(0, 2, 1)

        # 1b. Get query matrix (Q)
        if self._att_module == 'target':
            Q = self.Q
        elif self._att_module == 'label':
            Q = self.attention_layer._mapping_layer(self.Q.permute(1, 0)).permute(1, 0)
        elif self._att_module == 'self':
            Q = self.Q_1Dlayer(H).permute(0, 2, 1)
        elif self._att_module == 'alternate':
            Q1 = self.Q1
            Q2 = self.Q2
        elif self._att_module == 'hierarchical_target':
            Q1 = self.Q1
            Q2 = self.Q2
            if self._multihead is not True:
                Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)  # need to create query matrix for each doc
        elif self._att_module == 'hierarchical_label':
            Q1 = self.attention_layer._mapping_layer(self.Q1.permute(1, 0)).permute(1, 0)
            Q2 = self.attention_layer._mapping_layer(self.Q2.permute(1, 0)).permute(1, 0)
            if self._multihead is not True:
                Q2 = torch.unsqueeze(Q2, dim=0).repeat(K.size()[0], 1, 1)  # need to create query matrix for each doc

        # 3. Perform attention
        if self._multihead:
            if self._att_module != 'self':
                # For multihead attention we need to add batch_size to query matrices
                # The same query matrix is used for each sample in the batch
                if (self._att_module == 'alternate'):
                    Q1 = torch.unsqueeze(Q1.T, dim=0).repeat(K.size()[0], 1, 1)
                    Q2 = torch.unsqueeze(Q2.T, dim=0).repeat(K.size()[0], 1, 1)
                elif (self._att_module == 'hierarchical_target') or (self._att_module == 'hierarchical_label'):
                    Q1 = torch.unsqueeze(Q1.T, dim=0).repeat(K.size()[0], 1, 1)
                    Q2 = torch.unsqueeze(Q2.T, dim=0).repeat(K.size()[0], 1, 1)
                else:
                    Q = torch.unsqueeze(Q.T, dim=0).repeat(K.size()[0], 1, 1)

            if (self._att_module == 'alternate') or \
                    (self._att_module == 'hierarchical_target') or (self._att_module == 'hierarchical_label'):
                # Create num_heads splits of input/weights
                K, V, Q1, Q2 = self.multihead_weights(K=K, V=V, Q1=Q1, Q2=Q2)
                C, A = self.attention_layer(K=K, V=V, Q1=Q1, Q2=Q2)  # Perform attention
            else:
                K, V, Q = self.multihead_weights(K=K, V=V, Q=Q)  # Create num_heads splits of input/weights
                C, A = self.attention_layer(K=K, V=V, Q=Q)  # Perform attention
            A = transpose_output(X=A, num_heads=self._num_heads)  # Undo splits
            C = transpose_output(X=C, num_heads=self._num_heads)  # Undo splits
            C = self.multihead_weights.W_o(C).permute(0, 2, 1)  # Map multi-head outputs to one output

        # Single-head attention
        else:
            if (self._att_module == 'alternate') or \
                    (self._att_module == 'hierarchical_target') or (self._att_module == 'hierarchical_label'):
                C, A = self.attention_layer(K=K, V=V, Q1=Q1, Q2=Q2)
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

    """
    def __init__(self,
                 num_labels: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 scale: bool = False,
                 multihead: bool = False):
        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead

        # Initialze query embedding matrix using linear layer
        self.Q = nn.Linear(in_features=self._latent_doc_dim,
                           out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self,
                K: Union[torch.Tensor, List[torch.Tensor]],
                V: Union[torch.Tensor, List[torch.Tensor]],
                Q: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism

        Parameters
        ----------
        K : torch.Tensor
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : torch.Tensor
            Value embedding matrix - V ∈ R^lxd
        Q : torch.Tensor
            Query embedding matrix - Q ∈ R^nxd; where n: number of labels

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
            A = F.softmax(input=E, dim=2)
            # Compute context vector matrix C - dot product of attention matrix A and value embedding matrix V(H): QV.T
            # Where c_i represents the document context vector for the i-th label in the label space
            # C ∈ R^nxd, where n: number of labels and d: latent document dimension
            C = A.matmul(V)
        return C, A


class LabelAttention(nn.Module):
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

    """

    def __init__(self,
                 num_labels: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 label_embedding_matrix: np.array,
                 scale: bool = False,
                 multihead: bool = False):
        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead

        # Init label embedding matrix by using linear layer
        # Q ∈ R^nxd_e where n: number of labels in |L| and d_e: embedding dimension of tokens
        self.Q = nn.Linear(in_features=self._embedding_dim,
                           out_features=self._num_labels,
                           bias=True)
        self.Q.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Need a 1D-Conv layer to map the embedded (with embedding dimension) code descriptions
        # to the output dimension of the parallel convolution layers
        # See (Liu et al. 2021 - EffectiveCAN - 3.3 Attention)
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._latent_doc_dim,
                                        kernel_size=1)

    def forward(self,
                K: Union[torch.Tensor, List[torch.Tensor]],
                V: Union[torch.Tensor, List[torch.Tensor]],
                Q: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Forward pass of label-attention mechanism with pretrained label embedding matrix.

        Parameters
        ----------
        K : torch.Tensor
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : torch.Tensor
            Value embedding matrix - V ∈ R^lxd
        Q : torch.Tensor
            Query embedding matrix - Q ∈ R^nxd; where n: number of labels

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
            if self._scale:
                E = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(Q, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)
        else:
            if self._scale:
                E = Q.matmul(K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = Q.matmul(K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = A.matmul(V)
        return C, A


class SelfAttention(nn.Module):
    """
    This class performs self-attention with trainable weight matrices for the Key, Query, and Value.

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
                 multihead: bool = False):

        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead

    def forward(self,
                K: Union[torch.Tensor, List[torch.Tensor]],
                V: Union[torch.Tensor, List[torch.Tensor]],
                Q: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Forward pass of target attention mechanism
        Parameters
        ----------
        K : torch.Tensor
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : torch.Tensor
            Value embedding matrix - V ∈ R^lxd
        Q : torch.Tensor
            Query embedding matrix - Q ∈ R^lxd

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
            if self._scale:
                E = torch.bmm(Q, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
            else:
                E = torch.bmm(Q, K.permute(0, 2, 1))
            A = F.softmax(input=E, dim=-1)
            C = torch.bmm(A, V)
        else:
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
                 multihead: bool = False):

        super().__init__()
        self._num_labels = num_labels
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._scale = scale
        self._multihead = multihead

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

    def forward(self,
                K: Union[torch.Tensor, List[torch.Tensor]],
                V: Union[torch.Tensor, List[torch.Tensor]],
                Q1: Union[torch.Tensor, List[torch.Tensor]],
                Q2: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Forward pass of alternate attention mechanism
        Parameters
        ----------
        K : Union[torch.Tensor, List[torch.Tensor]]
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : Union[torch.Tensor, List[torch.Tensor]]
            Value embedding matrix - V ∈ R^lxd
        Q1 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^nxd
        Q2 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^nxd

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
            if self._scale:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)
                E2 = torch.bmm(Q2, K.permute(0, 2, 1)) / np.sqrt(self._embedding_dim)

            else:
                E1 = torch.bmm(Q1, K.permute(0, 2, 1))
                E2 = torch.bmm(Q2, K.permute(0, 2, 1))

            A1 = F.softmax(input=E1, dim=-1)
            A2 = F.softmax(input=E2, dim=-1)

            # Create mask to set alternating elements in AM and AN to 0
            mask_1 = torch.randn(A1.size()).bool()
            for i in range(A1.size()[-1]):
                if i % 2 != 0:
                    mask_1[:, :, i] = False
            A1 = A1.mul(mask_1)  # set every second element to 0 starting at index 1

            mask_2 = torch.randn(A2.size()).bool()
            for i in range(A2.size()[-1]):
                if i % 2 == 0:
                    mask_2[:, :, i] = False
            A2 = A2.mul(mask_2)  # set every second element to 0 starting at index 0

            # Combine AM and AN
            A = F.relu(A1 + A2)
            C = torch.bmm(A, V)
        else:
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
            mask_1 = torch.randn(A1.size()).bool()
            for i in range(A1.size()[-1]):
                if i % 2 != 0:
                    mask_1[:, :, i] = False
            A1 = A1.mul(mask_1)  # set every second element to 0 starting at index 1

            mask_2 = torch.randn(A2.size()).bool()
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
    """

    def __init__(self,
                 num_labels: int,
                 num_cats: int,
                 embedding_dim: int,
                 latent_doc_dim: int,
                 code2cat_map: List[int],
                 scale: bool = False,
                 multihead: bool = False):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
        self._scale = scale
        self._multihead = multihead

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_cats)
        nn.init.xavier_uniform_(self.Q1.weight)
        #        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)

        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_labels)
        nn.init.xavier_uniform_(self.Q2.weight)

    def forward(self,
                K: torch.Tensor,
                V: torch.Tensor,
                Q1: torch.Tensor,
                Q2: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

        Parameters
        ----------
        K : Union[torch.Tensor, List[torch.Tensor]]
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : Union[torch.Tensor, List[torch.Tensor]]
            Value embedding matrix - V ∈ R^lxd
        Q1 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^n1xd; where n1: number of categories in hierarchy level 1
        Q2 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^n2xd; where n2: number of codes in hierarchy level 2

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
                 multihead: bool = False):

        super().__init__()
        self._num_labels = num_labels
        self._num_cats = num_cats
        self._embedding_dim = embedding_dim
        self._latent_doc_dim = latent_doc_dim
        self._code2cat_map = code2cat_map
        self._scale = scale
        self._multihead = multihead

        # Initialize query matricees for hierarchical attention
        # Level 1: high-level category labels
        # Q1 ∈ R^n1xd where n1: number of labels of high-level categories and d: dim of latent doc representation
        self.Q1 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_cats)
        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)
        #        self.Q1.weight.data = torch.tensor(cat_embedding_matrix, dtype=torch.float)

        # Level 2: low-level code labels
        # Q_codes ∈ R^n2xd where n2: number of labels of low-level codes and d: dim of latent doc representation
        self.Q2 = nn.Linear(in_features=self._latent_doc_dim,
                            out_features=self._num_labels)
        self.Q2.weight.data = torch.tensor(label_embedding_matrix, dtype=torch.float)

        # Conv1D layer to avoid dimensionality mismatch
        self._mapping_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                        out_channels=self._latent_doc_dim,
                                        kernel_size=1)

    def forward(self,
                K: torch.Tensor,
                V: torch.Tensor,
                Q1: torch.Tensor,
                Q2: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of hierarchical target attention mechanism.
        This version uses the context matrix (C1) of hierarchy level 1 to inform the query matrix (Q2) of hierarchy
        level 2 by adding the created context vector C1_i for the ith category to the appropriate query embedding q2_i
        of the ith code description using a generated index mapping. In that, the codes are mapped to the categories.

        Parameters
        ----------
        K : Union[torch.Tensor, List[torch.Tensor]]
            Key embedding matrix - K ∈ R^lxd; where l: sequence length and d: latent document dimension
        V : Union[torch.Tensor, List[torch.Tensor]]
            Value embedding matrix - V ∈ R^lxd
        Q1 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^n1xd; where n1: number of categories in hierarchy level 1
        Q2 : Union[torch.Tensor, List[torch.Tensor]]
            Query embedding matrix - Q ∈ R^n2xd; where n2: number of codes in hierarchy level 2

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


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 key_dim: int,
                 query_dim: int,
                 value_dim: int,
                 hidden_dim: int,
                 att_module: str = None,
                 query_dim_cats: int = None,
                 bias: bool = False):
        super().__init__()
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._query_dim = query_dim
        self._value_dim = value_dim
        self._hidden_dim = hidden_dim
        self._att_module = att_module
        self._query_dim_cats = query_dim_cats
        self._bias = bias

        # Init multihead weights for keys, values, queries, and output weights
        # key embedding matrix
        self.W_k = nn.Linear(in_features=self._key_dim,
                             out_features=self._hidden_dim,
                             bias=self._bias)

        # query embedding matrix - codes
        if (self._att_module == 'hierarchical_target') or (self._att_module == 'hierarchical_label'):
            self.W_q = nn.Linear(in_features=self._query_dim_cats,
                                 out_features=self._hidden_dim,
                                 bias=self._bias)
        else:
            self.W_q = nn.Linear(in_features=self._query_dim,
                                 out_features=self._hidden_dim,
                                 bias=self._bias)
        # value embedding matrix
        self.W_v = nn.Linear(in_features=self._value_dim,
                             out_features=self._hidden_dim,
                             bias=self._bias)
        # output weights
        self.W_o = nn.Linear(in_features=self._hidden_dim,
                             out_features=self._query_dim,
                             bias=self._bias)

        if (self._att_module == 'alternate') or \
                (self._att_module == 'hierarchical_target') or (self._att_module == 'hierarchical_label'):
            self.W_q2 = nn.Linear(in_features=self._query_dim,
                                  out_features=self._hidden_dim,
                                  bias=self._bias)

    def forward(self, K: torch.Tensor, V: torch.Tensor, Q1: torch.Tensor, Q2: torch.Tensor = None) -> Tuple[torch.Tensor]:
        K = transpose_qkv(self.W_k(K), self._num_heads)
        V = transpose_qkv(self.W_v(V), self._num_heads)
        Q = transpose_qkv(self.W_q(Q1), self._num_heads)
        if Q2 is not None:
            Q2 = transpose_qkv(self.W_q2(Q2), self._num_heads)
            return K, V, Q, Q2
        return K, V, Q


# General utility functions
class AttentionInputs(nn.Module):
    """
    Class that computes the key, value, and query (self-attention) embedding matrices from the latent document
    representations after the CNN, LSTM, or Transformer.

    Parameters
    ----------
    latent_doc_dim : int
        Output dimension of encoder architecture, i.e., dimension of latent document representation
    """

    def __init__(self,
                 latent_doc_dim: int):
        super().__init__()
        self._latent_doc_dim = latent_doc_dim

        self._1Dconv = nn.Conv1d(in_channels=self._latent_doc_dim,
                                 out_channels=self._latent_doc_dim,
                                 kernel_size=1)
        nn.init.xavier_uniform_(self._1Dconv.weight)
        self._1Dconv.bias.data.fill_(0.01)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the key and value matrices.

        Parameters
        ----------
        H : torch.Tensor
            Latent representation of input sequence with shape: [batch_size: b, sequence_length: l, latent_dim: d]

        Returns
        -------
        M : torch.Tensor
            Embedding matrix

        """

        # Compute either key or value embedding matrix
        M = F.elu(self._1Dconv(H))
        return M


# Utility functions for multihead attention mechanism
def transpose_qkv(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Transposition for parallel computation of multiple attention heads.

    Parameters
    ----------
    X : torch.Tensor
        Input embedding matrix (Key, Value, Query)
    num_heads : int
        Number of attention heads

    Return
    ------
    torch.Tensor
        Transposed embedding matrix

    """
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    Reverse the operation of `transpose_qkv`

    Parameters
    ----------
    X : torch.Tensor
        Input embedding matrix (Key, Value, Query)
    num_heads : int
        Number of attention heads

    Return
    ------
    torch.Tensor
        Transposed embedding matrix

    """

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


