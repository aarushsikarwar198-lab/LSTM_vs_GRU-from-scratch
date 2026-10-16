# =============================================================================
# train.py
# What this file does:
#   Has the two functions that bench.py calls again and again:
#     - train_one_epoch(): trains the model for one full pass over the training data
#     - evaluate():        checks the model on val/test data WITHOUT learning
#   Both return (average loss, accuracy).
# =============================================================================

# (Classifier and time are imported here but not used in this file.)
from models import Classifier
import torch 
from torch import nn
import time 

def train_one_epoch(model , loader , optimizer , loss_fn , device):
    # running totals: sum of losses, number correct, number of reviews seen
    total_loss = 0.0
    correct = 0
    n = 0
    # training mode (turns dropout ON)
    model.train()
    # go through the data one batch at a time
    for tokens , labels in loader:
       # move the batch to the GPU (or CPU); tokens must be integers, labels must be floats
       tokens = tokens.to(device).long()
       labels = labels.to(device).float()
       # clear the gradients left over from the previous batch
       optimizer.zero_grad()
       # forward pass: model gives one score per review
       logits = model(tokens)
       # shape (batch, 1) -> (batch,)
       logits = logits.squeeze(1)
       # how wrong was the model on this batch?
       loss = loss_fn(logits, labels)
       # backward pass: find how to change each weight to reduce the loss
       loss.backward()
       # stop gradients from getting too large (max total size = 1.0)
       torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0)
       # update the weights
       optimizer.step()
       # add up the loss (multiplied by batch size so every review counts equally)
       total_loss+= loss.item()*labels.size(0)
       # score > 0 means "positive"; count how many predictions match the real labels
       correct += ((logits > 0).float()== labels).sum().item()
       n += labels.size(0)
    # average loss and accuracy over the whole epoch
    return total_loss/n , correct /n

# @torch.no_grad() = do not track gradients here (saves memory and time, we are not learning)
@torch.no_grad()
def evaluate(model , loader , loss_fn , device):
    # evaluation mode (turns dropout OFF)
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for tokens, labels in loader:
        tokens = tokens.to(device).long()
        labels = labels.to(device).float()
        logits = model(tokens)
        logits = logits.squeeze(1)
        loss = loss_fn(logits , labels)
        total_loss+= loss.item()*labels.size(0)
        correct += ((logits > 0).float()== labels).sum().item()
        n += labels.size(0)
    # average loss and accuracy over the whole loader
    return total_loss/n , correct/n