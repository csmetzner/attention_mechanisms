"""
This file contains source code for the training procedure of the models.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/24/2022
"""

# built-in libraries
import time
from typing import Dict, Union, List

# installed libraries
import torch
import torch.nn as nn
import numpy as np

# Custom libraries
from .performance_metrics import get_scores

# Select GPU as hardware if available otherwise use available CPU
device = ('cuda' if torch.cuda.is_available() else 'cpu')


def train(model: nn.Module,
          train_kwargs: Dict[str, Union[bool, int]],
          optimizer,
          train_loader,
          transformer: bool = False,
          val_loader=None,
          scheduler=None,
          save_name: str = None):
    """
    This function handles training and validating the model using the given training and validation datasets.

    Parameters
    ----------
    scheduler
    model : nn.Model
        Multi-label or multi-class classification model implemented in pytorch using nn.Model
    train_kwargs : Dict[str, Union[bool, int]]
        Dictionary storing all variables required to run training/validating process.
    optimizer : pytorch optimizer
        Optimizer used for controlling parameter training via backward propagation
    train_loader : pytorch data loader
        Dataloader containing the training dataset; samples X and ground-truth values Y
    transformer : bool; default=False
        Flag indicating whether the model is a transformer or not
    val_loader : pytorch data loader
        Dataloader contianing the validation dataset; samples X and ground-truth values Y
    scheduler : pytorch learning rate scheduler
        Learning rate scheduler
    save_name : str; default=None
        Descriptive name to save the trained model

    """
    epochs = train_kwargs['epochs']
    patience = train_kwargs['patience']

    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # Multilabel requires using the sigmoid function to compute pseudo-probabilities ranging [0, 1] for each label
    # Use BCEWithLogitsLoss() for increased numerical stability
    loss_fct = torch.nn.BCEWithLogitsLoss()

    # Variables to track validation performance and early stopping
    best_val_loss = np.inf
    patience_counter = 0

    ### Train model ###
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}', flush=True)
        # Enable training of layers with trainable parameters
        model.train()

        # Keep track of training time
        start_time = time.time()
        for b, batch in enumerate(train_loader):
            ## if-statement for debugging the code
            #if b == 1:
            #    break
            # set gradients to zero for every new batch
            optimizer.zero_grad()

            # Compute logits and return attention/energy scores if prompted
            if transformer:
                # cast input_ids and attention_mask to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask)
                Y = batch['labels'].to(device)
            else:
                # Cast samples to device; token2id mapped and 0-padded documents of current batch
                X = batch['X'].to(device)
                # Cast ground-truth labels to device; multi-label or multi-class tensors
                # Multi-label: multi-hot vectors; multi-class: class indices (tensor([0,1,2,3,4,5])
                Y = batch['Y'].to(device)
                logits = model(X)
            # Compute loss
            loss = 0
            loss += loss_fct(logits, Y)
            # perform backpropagation
            loss.backward()
            optimizer.step()
            l_cpu = loss.cpu().detach().numpy()

        scheduler.step()
        print(f'Training loss: {l_cpu} ({time.time() - start_time:.2f} sec)', flush=True)

        ### Validate model ###
        if val_loader is not None:
            scores = scoring(model=model,
                             data_loader=val_loader,
                             transformer=transformer,
                             quartiles_indices=None,
                             individual=False)
            val_loss = scores['loss']

            ### Early stopping to prevent overfitting ###
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{save_name}.pt')
            else:
                patience_counter += 1
                print(f'Patience: {patience_counter}')
                if patience_counter >= patience:
                    break
    # If no validation dataset available, save after every epoch
        else:
            torch.save(model.state_dict(), f'{save_name}.pt')


def scoring(model,
            data_loader,
            transformer: bool = False,
            quartiles_indices: List[int] = None,
            individual: bool = False,
            return_att_scores: bool = False) -> Dict[str, Union[float, np.array]]:

    """
    Parameters
    ----------

    model : nn.Model
        Multi-label or multi-class classification model implemented in pytorch using nn.Model
    data_loader : pytorch data loader
        Dataloader containing the training dataset; samples X and ground-truth values Y
    transformer : bool; default=False
        Flag indicating if model is a transformer or not.
    quartiles_indices : List[int]; default=None
        List containing information in which quartile a respective label is
    individual : bool; default=False
        Flag indicating if performance metrics should be computed for each label in the label space individually
    return_att_scores : bool; default=False
        Flag indicating if attention and energy scores should be retrieved

    Returns
    -------
    Dict[str, Union[float, np.array]]
        Dictionary containing the computed performance metrics (scores) and ground-truth values, prediction probabilites
        prediction, and validating/testing loss.
    """

    # Multilabel requires using the sigmoid function to compute pseudo-probabilities ranging [0, 1] for each label
    # Use BCEWithLogitsLoss() for increased numerical stability
    loss_fct = torch.nn.BCEWithLogitsLoss()

    # Put model in evaluation mode; turns off stochastic based layers (e.g., dropout or batch normalization)
    model.eval()

    # Init arrays to keep track of ground-truth labels and predictions
    y_trues = []
    y_probs = []
    y_preds = []

    # Init list to keep track of losses per batch and running validation loss variable
    losses = []

    # If return attention scores true then init empty arrays to collect attention and energy scores for the test data
    # lists store scores batch-wise; need to also store document indices to re-assign documents to scores
    if return_att_scores:
        attention_scores = []
        energy_scores = []

    # switch off autograd engine; reduces memory usage and increase computation speed
    with torch.no_grad():
        # loop through dataset
        for b, batch in enumerate(data_loader):
            # if statement for debugging the code
            #if b == 1:
            #    break
            if transformer:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                if return_att_scores:
                    logits, A, E = model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         return_att_scores=return_att_scores)
                    attention_scores.append(A.detach())
                    energy_scores.append(E.detach())
                else:
                    logits = model(input_ids=input_ids,
                                   attention_mask=attention_mask)
                Y = batch['labels'].to(device)
            else:
                # Retrieve token2id mapped and 0-padded documents of current batch
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)

                if return_att_scores:
                    logits, A, E = model(X, return_att_scores)
                    attention_scores.append(A.detach())
                    energy_scores.append(E.detach())
                else:
                    logits = model(X)
            loss = 0

            # Extend arrays with ground-truth values (Y), prediction probabilities (probs), and predictions (logits)
            y_trues.extend(Y.detach().cpu().numpy())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_probs.extend(probs)
            y_preds.extend(np.round(probs, 0))

            # Compute the loss for current batch
            loss += loss_fct(logits, Y)
            l_cpu = loss.cpu().detach().numpy()
            losses.append(l_cpu)

    # Compute the scores
    scores = {}
    scores, y_preds_, y_trues_, y_probs_ = get_scores(y_preds_=y_preds,
                                                      y_trues_=y_trues,
                                                      y_probs_=y_probs,
                                                      scores=scores,
                                                      ks=[5, 8, 15],
                                                      quartiles_indices=quartiles_indices,
                                                      individual=individual)

    loss = np.mean(losses)

    scores['y_trues'] = y_trues_
    scores['y_probs'] = y_probs_
    scores['y_preds'] = y_preds_
    scores['loss'] = loss

    # Store attention scores in score dictionary
    if return_att_scores:
        scores['attention_scores'] = attention_scores
        scores['energy_scores'] = energy_scores
    return scores
