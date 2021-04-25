import sys
sys.path.append('/home/nbuser/library/')

import os

import fastai
from fastai.basics import *
from fastai.text.all import *
from fastai.callback.all import *


from transformers import AutoModel, AutoTokenizer, AutoConfig
from .splitters import *
from .models import * 
from .navarin import * 

import torch

from math import floor
import random
import gc
import json

from datetime import datetime
from pprint import pprint

from sklearn.model_selection import StratifiedKFold

def augmentRep(rep):
  try:
    switcher = {
      # repertoire: (manteinance, generativity)
      "anticipazione": ('green',8,10),
      "causa": ('red',2,1),
      "commento": ('red',3,1),
      "conferma": ('red',2,1),
      "confronto": ('red',6,4), #= contrapposizione?
      "considerazione": ('green',3,5),
      "conseguenza": ('red',9,7), #= implicazione?
      "contrapposizione": ('red',6,4),
      "descrizione": ('green',0,1),
      "dichiarazione di intenti": ('red',5,4),
      "dichiarazione_intenti": ('red',5,4),
      "generalizzazione": ('red',3,1),
      "giudizio": ('red',3,1),
      "giustificazione": ('red',4,1),
      "implicazione": ('red',9,7),
      "non risposta": ('red',3,1),
      "non_risposta": ('red',3,1),
      "opinione": ('red',5,3),
      "possibilita": ('yellow',2,2),
      "prescrizione": ('yellow',4,4),
      "previsione": ('red',5,4),
      "proposta": ('green',4,5),
      "riferimento obiettivo": ('green',5,6),
      "riferimento_obiettivo": ('green',5,6),
      "sancire": ('red',1,0),
      "ridimensionamento": ('red',1,0), #=sancire?
      "specificazione": ('yellow',1,1),
      "valutazione": ('yellow',5,5),
    }
    return switcher[rep]
  except:
    return ('',0,0)

def assignPP(rep):
  try: 
    switcher = {
      "sancire": "A",
      "ridimensionamento": "A", #=sancire?
      "descrizione": "B",
      "specificazione": "A B C",
      "possibilita": "A B",
      "opinione": "A B C D E F",
      "riferimento_obiettivo": "A B C D G H",
      "causa": "A B I K",
      "conferma": "A B C L",
      "contrapposizione": "A B C D I E F",
      "confronto": "A B C D I E F", #= contrapposizione?
      "implicazione": "A B C D F I K H",
      "conseguenza": "A B C D F I K H", #= implicazione?
      "giudizio": "A B C F",
      "previsione": "A B C D I K H",
      "giustificazione": "A B C F K M",
      "non_risposta": "A B C F",
      "commento": "A B C F",
      "generalizzazione": "A B C F",
      "valutazione": "A C D F G O",
      "dichiarazione_intenti": "A B C D H F",
      "proposta": "A B C D G",
      "deresponsabilizzazione": "A B C F",
      "prescrizione": "A B C D G O",
      "considerazione": "A B C D G O",
      "anticipazione": "A B C D G H O"
    }
    return switcher[rep]
  except:
    return ""  


def cutoff_clean(df, max_seq_len, fai_tokenizer, tokenizer_vocab_ls):
  """ Cleans df of texts with > tokens than max_seq_len """
  
  df['is_valid'] = False
  splits = ColSplitter()(df)

  x_tfms = [attrgetter("text"), fai_tokenizer, Numericalize(vocab=tokenizer_vocab_ls)]
  dsets = Datasets(df, splits=splits, tfms=[x_tfms, [attrgetter("label"), Categorize()]], dl_type=SortedDL)

  #Factory method
  #dsets = TextDataLoaders.from_df(df, text_col="text", tok_tfm=fai_tokenizer, text_vocab=tokenizer_vocab_ls)

  trainItems = dsets.train.items['text_length'] <= max_seq_len
  validItems = dsets.valid.items['text_length'] <= max_seq_len


  df2 = df[pd.concat([trainItems,validItems])].copy().reset_index().drop(columns=['index'])


  return df2

def t_v_t_split(df,subsample_pct=1.0):
  """ Legacy method for a random train/valid/test split """

  df.sort_values(by=['label'],inplace=True)

  valid_pct = 0.20
  test_pct = 0.25

  valid_fct = floor((1 / valid_pct))
  test_fct = floor((1 / test_pct))

  train_tag, valid_tag, test_tag = (0,1,2)

  labels = df['label'].unique()

  tags = list()

  random.seed(42)

  for label in labels:
    lab_df = df[df["label"] == label]
    lab_tot = len(lab_df.index)

    if (subsample_pct < 1):
      lab_tot = int(floor(lab_tot*subsample_pct))
      lab_rest = len(lab_df.index) - lab_tot

    val_tot = lab_tot//valid_fct
    test_tot = lab_tot//test_fct
    train_tot = lab_tot - val_tot - test_tot
    lab_tags = [train_tag]*(train_tot) + [valid_tag]*(val_tot) + [test_tag]*(test_tot)


    if (subsample_pct < 1):
      lab_tags = lab_tags + [None]*lab_rest

    random.shuffle(lab_tags)
    tags = tags + lab_tags

  df['set_tag'] = tags

  df['is_valid'] = list(map(lambda x: True if (x == valid_tag) else False, tags))
  df['is_test'] = list(map(lambda x: True if (x == test_tag) else False, tags))

  if (subsample_pct < 1):
    df.dropna(subset=['set_tag'], inplace=True)

def set_potentials(df):
  df['p'] = df['label2'].apply(augmentRep)

def rep_label(df):  
  set_potentials(df)
  df['label'] = df['p'].apply(lambda x: x[0])

def get_test_df(df):
  test_df = df[df.is_test == True].copy().reset_index().drop(columns=['index'])
  df = df[df.is_test != True].copy().reset_index().drop(columns=['index'])

  return (df,test_df)
