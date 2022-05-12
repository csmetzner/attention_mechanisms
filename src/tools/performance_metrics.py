"""
This function contains source code for custom performance metric scores following the published code by:
    - Mullenbach et al. 2018 (https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
    - Vu et al. 2020 (https://github.com/aehrc/LAAT/blob/master/src/util/util.py)

Performance metrics:
    - F1-score micro/macro based on Mullenbach et al. 2018
    - F1-score micro/macro based on Sklearn
    - Precision @ k
    - AUC micro/macro

Author: Christoph Metzner
"""

# built-in libraries
from typing import List, Dict

# installed libraries
import numpy as np
from sklearn.utils import column_or_1d
from sklearn.metrics import roc_curve, auc, f1_score

###############################
##### Performance Metrics #####
###############################


def get_scores(y_preds_: List[np.array],
               y_trues_: List[np.array],
               y_probs_: List[np.array],
               scores: Dict[str, float],
               ks: List[int] = [5, 8, 15]) -> Dict[str, float]:
    """
    Function that retrieves scores
    Parameters
    ----------
    y_preds_ : List[np.array]
    y_trues_ : List[np.array]
    y_probs_ : List[np.array]
    ks : List[int]; default=[5, 8, 15]

    Returns
    -------
    Dict[str, float]
        Dictionary containing performance metrics

    """
    y_preds_ = np.array(y_preds_)
    y_trues_ = np.array(y_trues_)
    y_probs_ = np.array(y_probs_)

    # Compute F1-scores
    f1_macro_sk = f1_score(y_true=y_trues_, y_pred=y_preds_, average='macro')
    scores['f1_macro_sk'] = f1_macro_sk
    f1_micro_sk = f1_score(y_true=y_trues_, y_pred=y_preds_, average='micro')
    scores['f1_micro_sk'] = f1_micro_sk

    # Compute AUC-scores
    auc_scores = auc_metrics(y_probs=y_probs_, y_trues=y_trues_, y_trues_micro=y_trues_.ravel())
    scores['auc_macro'] = auc_scores['auc_macro']
    scores['auc_micro'] = auc_scores['auc_micro']

    # Compute Precision@k-scores
    for k in ks:
        prec_at_k = precision_at_k(y_trues=y_trues_, y_probs=y_probs_, k=k, pos_label=1)
        scores[f'prec@{k}'] = prec_at_k

    return scores, y_preds_, y_trues_, y_probs_


def macro_precision(y_preds: np.array, y_trues: np.array) -> float:
    """
    Function that computes the macro-averaged precision score for given predictions and ground-truth labels.
    Precision = true_positives / (true_positives + false_positives)

    Parameters
    ----------
    y_preds : np.array
        Binary prediction matrix
    y_trues : np.array
        Binary ground-truth matrix

    Returns
    -------
    float
        Macro-average precision score
    """

    prec = intersect_size(y_preds, y_trues, 0) / (y_preds.sum(axis=0) + 1e-10)
    return np.mean(prec)


def macro_recall(y_preds: np.array, y_trues: np.array) -> float:
    """
    Function that computes the macro-averaged recall score for given predictions and ground-truth labels.
    Recall = true_positives / (true_positives + false_negatives)

    Parameters
    ----------
    y_preds : np.array
        Binary prediction matrix
    y_trues : np.array
        Binary ground-truth matrix

    Returns
    -------
    float
        Macro-averaged recall score
    """

    rec = intersect_size(y_preds, y_trues, 0) / (y_trues.sum(axis=0) + 1e-10)
    return np.mean(rec)


def macro_f1(y_preds, y_trues) -> float:
    """
    Function that computes the macro-averaged F1-score.

    Parameters
    ----------
    y_preds : np.array
        Binary prediction matrix
    y_trues : np.array
        Binary ground-truth matrix

    Returns
    -------
    float
        Macro-average f1-score

    """

    prec = macro_precision(y_preds=y_preds, y_trues=y_trues)
    rec = macro_recall(y_preds=y_preds, y_trues=y_trues)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def micro_precision(y_preds_mic: np.array, y_trues_mic: np.array) -> float:
    """
    Function that computes the micro-averaged precision score.

    Parameters
    ----------
    y_preds_mic : np.array
        Binary prediction matrix, raveled
    y_trues_mic : np.array
        Binary ground-truth matrix, raveled

    Returns
    -------
    float
        Micro-averaged precision score

    """
    if y_preds_mic.sum(axis=0) == 0:
        return 0.0
    return intersect_size(y_preds_mic, y_trues_mic, 0) / y_preds_mic.sum(axis=0)


def micro_recall(y_preds_mic: np.array, y_trues_mic: np.array) -> float:
    """
    Function that computes the micro-averaged recall score.

    Parameters
    ----------
    y_preds_mic : np.array
        Binary prediction matrix, raveled
    y_trues_mic : np.array
        Binary ground-truth matrix, raveled

    Returns
    -------
    float
        Micro-averaged recall score

    """
    if y_trues_mic.sum(axis=0) == 0:
        return 0.0
    return intersect_size(y_preds_mic, y_trues_mic, 0) / y_trues_mic.sum(axis=0)


def micro_f1(y_preds_mic, y_trues_mic):
    """
    Function that computes the macro-averaged F1-score.

    Parameters
    ----------
    y_preds : np.array
        Binary prediction matrix
    y_trues : np.array
        Binary ground-truth matrix

    Returns
    -------
    float
        Macro-average f1-score

    """
    prec = micro_precision(y_preds_mic, y_trues_mic)
    rec = micro_recall(y_preds_mic, y_trues_mic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


##############################
###### Helper functions ######
##############################


def intersect_size(y_preds: np.array, y_trues: np.array, axis: int=0) -> List[float]:
    """
    Function that retrieves true positives per label.

    Parameters
    ----------
    y_preds : np.array
        Binary prediction matrix
    y_trues : np.array
        Binary ground-truth matrix
    axis : int; default=0
        Indicates dimension to be summed over, i.e., sums true positives per label

    Returns
    -------
    List[float]
        List containing number of true positives per label
    """
    return np.logical_and(y_preds, y_trues).sum(axis=axis).astype(float)


def precision_at_k(y_trues: np.array,
                   y_probs: np.array,
                   k: int,
                   pos_label: int = 1) -> float:
    """
    Function that computes the precision at k across a dataset of documents.

    Parameters
    ----------
    y_trues : np.array [n_docs, n_labels]
        Array containing the ground-truth labels for each document in the dataset
    y_probs : np.array [n_docs, n_labels]
        Array containing the prediction scores (probabilities) of each label for each document in the dataset
    k : int
        number of labels with largest confidence considered to compute precision
    pos_label : int; default=1
        Integer indicating which binary label is class of importance

    Returns
    -------
    float
        Prec@k for considered k
    """

    # Compute Prec@k for each k in list
    prec_at_k_scores = []
    # Loop through ground-truth and prediction values for each document in the dataset
    for y_true, y_prob in zip(y_trues, y_probs):
        # transform torch.Tensor to numpy array
        y_true_arr = column_or_1d(y_true)
        y_prob_arr = column_or_1d(y_prob)

        # Get boolean values to sort
        y_true_arr = (y_true_arr == pos_label)

        # Sort arrays
        desc_sort_order = np.argsort(y_prob_arr)[::-1]
        y_true_sorted = y_true_arr[desc_sort_order]

        # count number of true_positives
        true_positives = y_true_sorted[:k].sum()
        prec_at_k_scores.append(true_positives / k)
    # Return average Prec@k across all documents in dataset
    return np.mean(prec_at_k_scores)


def auc_metrics(y_probs: np.array, y_trues: np.array, y_trues_micro: np.array) -> Dict[str, float]:
    """
    Function that computes the micro and macro scores for the area-under-curve (AUC) score. Code taken from Mullenbach
    et al. 2018 to enforce reproducibility and comparability of metric.
    Source: https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py  - Line: 169

    Parameters
    ----------
    y_probs : np.array
        Prediction scores (probabilities) for each label prediction
    y_trues : np.array
        Ground truth values per document
    y_trues_micro : np.array
        Ground truth values of all documents in one array and not per document

    Returns
    -------
    Dict[str, float]
        Micro and macro AUC scores

    """

    if y_probs.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y_trues.shape[1]):
        #only if there are true positives for this label
        if y_trues[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y_trues[:, i], y_probs[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = y_probs.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_trues_micro, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc
