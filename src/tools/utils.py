"""
This file contains source code for utility functions.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/03/2022
"""

# installed libraries
import numpy as np
from sklearn.utils import class_weight


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
