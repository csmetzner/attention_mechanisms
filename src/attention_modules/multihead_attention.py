"""
The file contains the code for the implementation of the multi-head attention mechanism
(source: https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html).
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 06/01/2022
    @last modified: 03/07/2023

"""
# installed libraries
import torch

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
