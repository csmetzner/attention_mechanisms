"""
Source code that contains a basic convolution neural network architecture with three parallel convolution layers.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/14/2022
    @last modified: 05/24/2022
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
            self.transformer_model = AutoModel.from_pretrained('/home/u0z/attention_mechanisms/src/models/Clinical-Longformer/')
            self._latent_doc_dim = 768

        # Init dropout layer
        self.dropout_layer = nn.Dropout(p=self._dropout_p)

        # Init Attention Layer
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
                attention_mask: torch.Tensor,
                return_doc_embeds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass of transformer model

        Parameters
        ----------
        input_ids : torch.Tensor
            Token indices, numerical representation of tokens building the sequences that are used as input by the model
        token_type_ids : torch.Tensor
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        attention_mask : torch.Tensor
            This mask indicates the model which tokens should paid attention to or not. Will ignore padding tokens.
        return_doc_embeds : bool; default=False
            Flag indicating if doc embeddings should be returned

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor]]
            [Logits], [Logits, doc_embeds]
        """
        # Get latent document representation
        H = self.transformer_model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   return_dict=True,
                                   output_hidden_states=True)
        # retrieve hidden state of CLS token
        #H = H['hidden_states'][-1][:, 0, :].unsqueeze(dim=1).permute(0, 2, 1)
        H = F.relu(H['hidden_states'][-1].permute(0, 2, 1))

        H = self.dropout_layer(H)

        # Add attention module here
        C, att_scores = self.attention_layer(H=H)

        if self._att_module == 'self':
            # Necessary to match output with |L| ground-truth labels
            logits = self.output_layer(C).sum(dim=1)
        else:
            logits = self.output_layer(C).sum(dim=2)  # Consider .sum(dim=1) - depends on number of attention vectors

        if return_doc_embeds:
            return logits, H
        return logits
