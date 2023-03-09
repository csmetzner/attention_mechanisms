"""
Source code that contains a basic convolution neural network architecture with three parallel convolution layers.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/14/2022
    @last modified: 03/07/2023
"""

# built-in libraries
from typing import List, Tuple, Union

# installed libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModel

# custom libraries
from attention_modules.attention_mechanisms import Attention
device = ('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):
    """
    Convolution neural network class with three parallel convolution layers.

    Parameters
    ----------
    n_labels : int
        Number of labels considered in the label space
    embedding_dim : int
        Dimension of word embeddings, i.e., dense vector representation
    model_name : str
        Name of transformer model
            - 'DischargeBERT': BERT-based model pretrained on the MIMIC-III clinical notes using only discharge summary
            https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT
    att_module : str; default=None
        Defines the attention module/mechanism applied to perform attention to the latent document representation input
    scale : bool; default=False
        Flag indicating if energy scores (QxK.T) should be scaled by the root of
    multihead : bool; default=False
        Flag indicating if multi-head attention is used
    num_heads : int; default=None
        Number of attention heads when multi-head attention is activated
    n_cats : int
        Number of high-level categories to perform hierarchical attention
    label_embedding_matrix : np.array
        Embedding matrix pretrained on the code descriptions
    cat_embedding_matrix : np.array
        Embedding matrix pretrained on the category descriptions
    code2cat_map : List[int]; default=None
        Category index to map codes to categories

    """
    def __init__(self,
                 n_labels: int,
                 embedding_dim: int,
                 model_name: str,
                 att_module: str = 'target',
                 scale: bool = False,
                 multihead: bool = False,
                 num_heads: int = None,
                 n_cats: int = None,
                 label_embedding_matrix: np.array = None,
                 cat_embedding_matrix: np.array = None,
                 code2cat_map: List[int] = None,
                 hidden_size: int = 768,
                 dropout_p: float = 0.5,
                 embedding_scaling: float = 1):
        super().__init__()
        self._n_labels = n_labels
        self._embedding_dim = embedding_dim
        self._model_name = model_name
        self._att_module = att_module
        self._scale = scale
        self._multihead = multihead
        self._num_heads = num_heads
        self._n_cats = n_cats
        self._label_embedding_matrix = label_embedding_matrix
        self._cat_embedding_matrix = cat_embedding_matrix
        self._code2cat_map = code2cat_map
        self._dropout_p = dropout_p
        self._embedding_scaling = embedding_scaling

        if self._model_name == 'ClinicalLongformer':
            self.transformer_model = AutoModel.from_pretrained("/Users/cmetzner/Desktop/Study/PhD/research/ORNL/Biostatistics and Multiscale System Modeling/"
                                                               "attention_mechanisms/src/models/Clinical-Longformer")
            self._latent_doc_dim = 768

        # Init dropout layer
        self.dropout_layer = nn.Dropout(p=self._dropout_p)

        # Init Attention Layer
        if self._att_module != 'baseline':
            self.attention_layer = Attention(num_labels=self._n_labels,
                                             embedding_dim=self._embedding_dim,
                                             latent_doc_dim=self._latent_doc_dim,
                                             att_module=self._att_module,
                                             scale=self._scale,
                                             multihead=self._multihead,
                                             num_heads=self._num_heads,
                                             num_cats=self._n_cats,
                                             embedding_scaling=self._embedding_scaling,
                                             label_embedding_matrix=self._label_embedding_matrix,
                                             cat_embedding_matrix=self._cat_embedding_matrix,
                                             code2cat_map=self._code2cat_map)

        # Init output layer
        self.output_layer = nn.Linear(in_features=self._latent_doc_dim,
                                      out_features=self._n_labels)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self,
                input_ids: torch.Tensor,
                return_en_scores: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass of transformer model

        Parameters
        ----------
        input_ids : torch.Tensor
            Token indices, numerical representation of tokens building the sequences that are used as input by the model
        return_en_scores : bool; default=False
            Flag indicating to return attention and energy scores

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor]]
            [Logits], [Logits, doc_embeds]
        """
        # Get latent document representation
        H = self.transformer_model(input_ids=input_ids,
                                   return_dict=True,
                                   output_hidden_states=True)
        # Retrieve latent document representation of last layer in transformer model
        H = H['hidden_states'][-1]
        H = self.dropout_layer(H)  # [batch_size, sequence_len == 4096, hidden_dim == 768]

        if self._multihead:
            C = self.attention_layer(H=H.permute(0, 2, 1))  # [batch_size, num_labels, hidden_dim]
            logits = self.output_layer.weight.mul(C).sum(dim=2).add(self.output_layer.bias)
        else:
            if self._att_module == 'baseline':
                # The baseline approach of the clinical longformer uses the latent document representation of the first
                # token of the sequence [<s>]
                H = H[:, 0, :]  # [batch_size, hidden_dim]
                logits = self.output_layer(H)  # [batch_size, num_labels]

            elif self._att_module == 'target':
                # target attention uses a one query vector to learn a single latent document representation
                C, E = self.attention_layer(H=H.permute(0, 2, 1))  # [batch_size, 1, hidden_dim]
                logits = self.output_layer(C)  # [batch_size, 1, num_labels]
                logits = torch.squeeze(logits, dim=1)  # [batch_size, num_labels]
            else:
                C, E = self.attention_layer(H=H.permute(0, 2, 1))  # [batch_size, num_labels, hidden_dim]
                logits = self.output_layer.weight.mul(C).sum(dim=2).add(self.output_layer.bias)  # [batch_size, num_labels]

        if return_en_scores:
            E_new = torch.zeros((E.size()[0], E.size()[1], 4096))  # 4096 - max document length for CLF
            E_new[:, :, :E.size()[2]] = E
            E_new = E_new.to(device)
            return logits, E_new
        return logits
