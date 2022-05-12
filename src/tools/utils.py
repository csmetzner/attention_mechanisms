"""
This file contains source code for utility functions.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/06/2022
"""
# built-in libraries
import os
import pickle

# installed libraries
import numpy as np
from sklearn.utils import class_weight
from gensim.models import Word2Vec


def get_class_weights(y_train) -> np.array:
    """
    This function computes the class weights for a given imbalanced dataset.

    Parameters
    ----------
    y_train : np.array
        Array of the classes occurring in the data, as given by np.unique(y_org) with y_org the original class labels.

    Returns
    -------
    class_weights : np.array of shape (n_classes,)
        Array with class_weight_vect[i]; the weight for i-th class.

    """
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)

    return class_weights


def get_word_embedding_matrix(dataset: str,
                              embedding_dim: int,
                              path_data: str,
                              min_count: int = 3) -> np.array:
    """
    Function that loads pre-trained word embedding matrix or creates a word embedding matrix for given word embedding.

    Parameters
    ----------
    dataset : str
        Selected dataset: | PathReports | MimicFull | Mimic50 |
    embedding_dim : int
        Dimension of dense vector representation (word embeddings) of each word
    path_data : str
        Location of directory where processed data is stored
    min_count : int
        Minimum word frequency in training corpus to be considered; words with lower frequency are considered to be
        unknown '<unk>'

    Returns
    -------
    np.array
        Pre-trained word embedding matrix with shape [n_words + 2, embedding_dim]; + 2 for <pad> and <unk>
    """

    path_dataset = os.path.join(path_data, f'data_{dataset}')
    path_embeddings = os.path.join(path_dataset, 'word_embeddings')

    # Create new folder to store word embedding matrices if none exists
    if not os.path.exists(path_embeddings):
        os.makedirs(path_embeddings)

    # Load existing word embedding matrix or create one for given embedding matrix
    path_matrix = os.path.join(path_embeddings, f'wv_{dataset}_{embedding_dim}.pkl')
    if os.path.isfile(path_matrix):
        with open(path_matrix, 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        # Load tokens in training split of dataset
        with open(os.path.join(path_dataset, f'train_tokens_{dataset}.pkl'), 'rb') as f:
            train_tokens = pickle.load(f)

        # Load created vocabulary based on training split of dataset
        with open(os.path.join(path_dataset, f'vocab_{dataset}.pkl'), 'rb') as f:
            vocab = pickle.load(f)

        # Create word embeddings using Word2Vec
        wv_w2v = Word2Vec(sentences=train_tokens, vector_size=embedding_dim, window=5, min_count=min_count, workers=4)

        # create empty word embedding matrix with size of vocabulary, i.e., increased by two ('<pad>', '<unk>')
        embedding_matrix = np.zeros((len(vocab), int(embedding_dim)))

        # Now add word embeddings created with Word2Vec to embedding_matrix
        # Making sure that the word embedding is placed on the correct index
        wv_w2v_vectors = wv_w2v.wv.vectors

        # since wv_w2v tokens have same order as pytorch implementation of build_vocab_from_iterator we can just
        # use proper indexing to populate embedding matrix; first two embeddings are empty - <pad> and <unk>
        embedding_matrix[2:] = wv_w2v_vectors

        # Store freshly created embedding_matrix
        with open(path_matrix, 'wb') as f:
            pickle.dump(embedding_matrix, f)

    return embedding_matrix


def parse_boolean(value: str) -> bool:
    """
    Function that helps with argparse arguments
    Parameters
    ----------
    value : str

    Returns
    -------
    bool
    """
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False
