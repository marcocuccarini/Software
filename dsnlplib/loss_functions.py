import sys
import os

import fastai
from fastai.basics import *
from fastai.text.all import *
from fastai.callback.all import *


from transformers import AutoModel, AutoTokenizer, AutoConfig
from .splitters import *
from .models import * 
from .navarin import * 
from .data_load import *

from .loss_functions import *
from .legacy_utils import *

import torch

from math import floor
import random
import gc
import json

from datetime import datetime
from pprint import pprint

from statistics import mean

from sklearn.model_selection import StratifiedKFold

def WeightedCrossEntropyLoss(dsc, df):
  n_classes = dsc.num_labels
  n_samples = len(df)
  weights = [n_samples / (n_classes * n_label_samples) for n_label_samples in list(df.groupby('label').size())]
  class_weights = torch.FloatTensor(weights).cuda()

  criterion = fastai.losses.CrossEntropyLossFlat(weight=class_weights)
  return criterion 


class WeightedLabelSmoothingCrossEntropy(Module):
    y_int = True
    def __init__(self, dsc, df, eps:float=0.1, reduction='mean'): 

      self.eps,self.reduction = eps,reduction
      n_classes = dsc.num_labels
      n_samples = df.size
      weights = [n_samples / (n_classes * n_label_samples) for n_label_samples in list(df.groupby('label').size())]
      self.weight = torch.FloatTensor(weights).cuda()


    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)

