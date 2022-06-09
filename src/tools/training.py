"""
This file contains source code for the training procedure of the models.
    @author: Christoph Metzner
    @email: cmetzner@vols.utk.edu
    @created: 05/03/2022
    @last modified: 05/24/2022
"""

# built-in libraries
import time
import random
from typing import Dict, Union, List

# installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom libraries
from .performance_metrics import get_scores

# Select GPU as hardware if available otherwise use available CPU
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model: nn.Module,
          train_kwargs: Dict[str, Union[bool, int]],
          device,
          optimizer,
          train_loader,
          transformer: bool,
          val_loader=None,
          class_weights: np.array = None,
          save_name: str = None,
          alignment_model = None):
    """
    This function handles training and validating the model using the given training and validation datasets.

    Parameters
    ----------
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
    class_weights : np.array; default=None
        Array containing the class weights in shape (n_classes,)
    save_name : str; default=None
        Descriptive name to save the trained model
    alignment_model: nn.Module; default=None
        Alignment model contains the critic and the navigator to determine the alignment loss.

    """

    epochs = train_kwargs['epochs']
    patience = train_kwargs['patience']
    multilabel = train_kwargs['multilabel']
    alignment = train_kwargs['alignment']

    # Set up loss function
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if multilabel:
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # Multilabel requires using the sigmoid function to compute pseudo-probabilities ranging [0, 1] for each label
        # Use BCEWithLogitsLoss() for increased numerical stability
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    else:
        # Multiclass - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Variables to track validation performance and early stopping
    best_val_loss = np.inf
    patience_counter = 0

    ### Train model ###
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}', flush=True)
        # Enable training of layers with trainable parameters
        model.train(True)

        # Init arrays to keep track of ground-truth labels
        y_trues = []
        y_preds = []

        # Keep track of training time
        start_time = time.time()
        for b, batch in enumerate(train_loader):
            # set gradients to zero for every new batch
            optimizer.zero_grad()
            if transformer:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
                Y = batch['labels'].to(device)
            else:
                # Cast samples to device; token2id mapped and 0-padded documents of current batch
                X = batch['X'].to(device)
                # Cast ground-truth labels to device; multi-label or multi-class tensors
                # Multi-label: multi-hot vectors; multi-class: class indices (tensor([0,1,2,3,4,5])
                Y = batch['Y'].to(device)
                logits = model(X)
            loss = 0
            #if alignment:
            #    alignment_model._optim_critic.zero_grad()
            #    alignment_model._optim_navigator.zero_grad()
            #    alignment_loss = alignment_model(K=model.module.attention_layer.attention_layer.K_alignment,
            #                                    Q=model.module.attention_layer.attention_layer.Q_alignment)
            #    loss = loss + alignment_loss
            #    alignment_loss.backward(retain_graph=True)
            #    alignment_model._optim_critic.step()
            #    alignment_model._optim_navigator.step()
            print(f'X.device; {X.device}')
            print(f'Y.device: {Y.device}')
            print(f'Logits.device: {logits.device}')

            y_trues.extend(Y.detach().cpu().numpy())
            y_preds.extend(logits.detach().cpu().numpy()) # how do you have to compute these things for multi-class case
            output = loss_fct(logits, Y)
            loss = loss + output
            print(f'loss: {loss.device}')
            print(f'output: {output.device}')
            # Perform backpropagation
            loss.backward()
            optimizer.step()

            l_cpu = loss.cpu().detach().numpy()
            #if b == 1:
            #   break
        print(f'Training loss: {l_cpu} ({time.time() - start_time:.2f} sec)', flush=True)


        ### Validate model ###
        if val_loader is not None:
            print('\nValidating current model with validation dataset.')
            scores = scoring(model=model,
                             data_loader=val_loader,
                             class_weights=class_weights,
                             multilabel=multilabel,
                             transformer=transformer)
            val_loss = scores['loss']

            ### Early stopping to prevent overfitting ###
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{save_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping! No improvement in validation performance!', flush=True)
                    break
    # If no validation dataset available, save after every epoch
        else:
            torch.save(model.state_dict(), f'{save_name}.pt')


def scoring(model,
            data_loader,
            multilabel: bool,
            device,
            transformer: bool = False,
            class_weights: np.array = None) -> Dict[str, Union[float, np.array]]:

    """
    Parameters
    ----------
    model : nn.Model
        Multi-label or multi-class classification model implemented in pytorch using nn.Model
    data_loader : pytorch data loader
        Dataloader containing the training dataset; samples X and ground-truth values Y
    multilabel : bool
        Flag indicating whether multi-label or multi-class text classification is taken place
    class_weights : np.array
        Array containing the class weights in shape (n_classes,)

    Returns
    -------
    Dict[str, Union[float, np.array]]
        Dictionary containing the computed performance metrics (scores) and ground-truth values, prediction probabilites
        prediction, and validating/testing loss.
    """

    # Set up loss function
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if multilabel:
        # Multilabel requires using the sigmoid function to compute pseudo-probabilities ranging [0, 1] for each label
        # Use BCEWithLogitsLoss() for increased numerical stability
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    else:
        # Multiclass
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Put model in evaluation mode; turns off stochastic based layers (e.g., dropout or batch normalization)
    model.eval()

    # Init arrays to keep track of ground-truth labels and predictions
    y_trues = []
    y_probs = []
    y_preds = []

    # Init list to keep track of losses per batch and running validation loss variable
    losses = []

    # switch off autograd engine; reduces memory usage and increase computation speed
    with torch.no_grad():
        # loop through dataset
        for b, batch in enumerate(data_loader):
            if transformer:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
                Y = batch['labels'].to(device)
            else:
                # Retrieve token2id mapped and 0-padded documents of current batch
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                logits = model(X)
            loss = 0
            print(f'X.device; {X.device}')
            print(f'Y.device: {Y.device}')
            print(f'Logits.device: {logits.device}')
            #print(f'loss.device: {loss.device}')


            # Extend arrays with ground-truth values (Y), prediction probabilities (probs), and predictions (logits)
            y_trues.extend(Y.detach().cpu().numpy())
            if multilabel:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                y_probs.extend(probs)
                y_preds.extend(np.round(probs, 0))
            else:
                soft_out = F.softmax(logits, dim=-1).max(-1)
                probs = soft_out[0].detach().cpu().numpy()
                probs_idx = soft_out[1].detach().cpu().numpy()
                y_probs.extend(probs)
                y_preds.extend(probs_idx)

            # Compute the loss for current batch
            loss += loss_fct(logits, Y)
            print(f'loss.device: {loss.device}')
            l_cpu = loss.cpu().detach().numpy()
            losses.append(l_cpu)
            #if b == 1:
            #    break

    # Compute the scores
    scores = {}
    scores, y_preds_, y_trues_, y_probs_ = get_scores(y_preds_=y_preds,
                                                      y_trues_=y_trues,
                                                      y_probs_=y_probs,
                                                      scores=scores,
                                                      ks=[5, 8, 15])

    loss = np.mean(losses)

    scores['y_trues'] = y_trues_
    scores['y_probs'] = y_probs_
    scores['y_preds'] = y_preds_
    scores['loss'] = loss

    return scores
