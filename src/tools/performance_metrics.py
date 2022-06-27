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
               ks: List[int] = [5, 8, 15],
               quartiles_indices: List[int] = None,
               individual: bool = False,) -> Dict[str, float]:
    """
    Function that retrieves scores
    Parameters
    ----------
    y_preds_ : List[np.array]
        Array containing predictions of validation/test dataset
    y_trues_ : List[np.array]
        Array containing ground truths of validation/test dataset
    y_probs_ : List[np.array]
        Array containing prediction scores (probabilities) of validation/test dataset
    ks : List[int]; default=[5, 8, 15]
        List of precision @ k values
    quartiles_indices: List[int]; default=None
        Flag indicating if performance metrics should be computed for the four quartiles
    individual: bool; default=False
        Flag indicating if performance metrics should be computed for each individual label

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

    # Check if performance for quartiles should be computed
    if quartiles_indices is not None:
        # Split predictions and ground-truths into quartiles
        preds_quartiles = {0: [], 1: [], 2: [], 3: []}
        trues_quartiles = {0: [], 1: [], 2: [], 3: []}

        # Nested for-loop to extract predictions and ground-truths for each quartiles per document
        for preds, trues in zip(y_preds_, y_trues_):
            # This preserves the original label structure of nested documents - used for F1-micro computation
            preds_quartiles_doc = {0: [], 1: [], 2: [], 3: []}
            trues_quartiles_doc = {0: [], 1: [], 2: [], 3: []}

            # quartiles_indices indicates the membership of a label to a specific quartile [0, 1, 2, 3]
            # We enumerate through the list of quartile assignments
            # index: refers to the current label
            # quart: refers to the quartile membership
            # 0: (-inf, Q1)
            # 1: (Q1, Q2)
            # 2: (Q2, Q3)
            # 3: (Q3, +inf)
            for index, quart in enumerate(quartiles_indices):
                preds_quartiles_doc[quart].append(preds[index])
                trues_quartiles_doc[quart].append(trues[index])

            # This for loop add the quartile lists per document to the overall quartiles
            for quartile_idx in range(4):
                preds_quartiles[quartile_idx].append(preds_quartiles_doc[quartile_idx])
                trues_quartiles[quartile_idx].append(trues_quartiles_doc[quartile_idx])

        for quartile_idx in range(4):
            f1_macro = f1_score(y_true=trues_quartiles[quartile_idx], y_pred=preds_quartiles[quartile_idx], average='macro')
            scores[f'f1_macro_Q{quartile_idx}'] = f1_macro
            f1_micro = f1_score(y_true=trues_quartiles[quartile_idx], y_pred=preds_quartiles[quartile_idx], average='micro')
            scores[f'f1_micro_Q{quartile_idx}'] = f1_micro

    # Check if performance for each individual label should be computed
    if individual:
        preds_individual = [ [] for _ in range(len(y_preds_[0]))]
        trues_individual = [ [] for _ in range(len(y_trues_[0]))]
        for preds, trues in zip(y_preds_, y_trues_):
            for i, (pred, true) in enumerate(zip(preds, trues)):
                preds_individual[i].append(pred)
                trues_individual[i].append(true)

        for i in range(len(y_preds_[0])):
            f1_micro = f1_score(y_true=trues_individual[i], y_pred=preds_individual[i], average='micro')
            scores[f'f1_micro_label{i}'] = f1_micro

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
