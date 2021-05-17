# -*- coding: utf-8 -*-
"""dsnlp-lib

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sx1BQQtw9Ti1sReJYNc5aELVdcrh3Rys
"""

import sys
sys.path.append('/home/nbuser/library/')

import os

import fastai
from fastai.basics import *
from fastai.text.all import *
from fastai.callback.all import *
from keras.models import model_from_json

import sklearn.linear_model as sk
from sklearn.multioutput import MultiOutputClassifier
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

import pandas as pd
import numpy as np
from functools import reduce

from IPython.display import display, HTML

import pickle
import copy 

from datetime import datetime
from pprint import pprint

from statistics import mean

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as skm

class DSTransform(Transform):
    def __init__(self,exp):
      self.exp = exp


     #codifica le frasi 
    def encodes(self, i):
        
        question = i.Question
        answer = i.Answer
        label = i.label

        input_ids, attention_mask, token_type_ids = self.exp.qa_tok_func((question,answer))

        #print(tokenized, flush=True)

        tokenized = torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)

        return (tokenized)

    def decodes(self, x): 
      return self.exp.tokenizer.decode(x[0])

    
#    def __len__(self): return len(self.exp.df)

#collapse
class TokFunc():
    """ 
        transformer_tokenizer : takes the tokenizer that has been loaded from the tokenizer class
        pretrain_id : model type set by the user
        max_seq_len : override default sequence length, typically 512 for bert-like models
        sentence_pair : whether a single sentence (sequence) or pair of sentences are used
    """
    def __init__(self, transformer_tokenizer=None, pretrain_id = 'roberta', max_seq_len=None, 
                 sentence_pair=False, **kwargs): 
        self.tok, self.max_seq_len=transformer_tokenizer, max_seq_len
        if self.max_seq_len:
            if self.max_seq_len > self.tok.model_max_length: 
                print('WARNING: max_seq_len is larger than the model default transformer_tokenizer.model_max_length')
        if sentence_pair: self.max_seq_len=ifnone(max_seq_len, self.tok.max_len_sentences_pair) 
        else: self.max_seq_len=ifnone(max_seq_len, self.tok.max_len_single_sentence)
        self.pretrain_id = pretrain_id
        
    def do_tokenize(self, o:str):
        """Limits the maximum sequence length and add the special tokens"""
        CLS, SEP=self.tok.cls_token, self.tok.sep_token
        
        # Add prefix space, depending on model selected
        if 'roberta' in self.pretrain_id: tokens=self.tok.tokenize(o, add_prefix_space=True)[:self.max_seq_len]
        else: tokens = self.tok.tokenize(o)[:self.max_seq_len]
        
        # order of 'tokens', 'SEP' and 'CLS'
        if 'xlnet' in self.pretrain_id: return tokens + [SEP] +  [CLS]
        else: return [CLS] + tokens + [SEP]

    def __call__(self, items): 
        for o in items: yield self.do_tokenize(o)

#collapse
class QATokFunc():
    """ 
        # TODO: MERGE WITH TokFunc
        transformer_tokenizer : takes the tokenizer that has been loaded from the tokenizer class
        pretrain_id : model type set by the user
        max_seq_len : override default sequence length, typically 512 for bert-like models
        sentence_pair : whether a single sentence (sequence) or pair of sentences are used
    """
    def __init__(self, transformer_tokenizer=None, pretrain_id = 'roberta', max_seq_len=None, 
                 sentence_pair=False, **kwargs): 
        self.tok, self.max_seq_len=transformer_tokenizer, max_seq_len
        if self.max_seq_len:
            if self.max_seq_len > self.tok.model_max_length: 
                print('WARNING: max_seq_len is larger than the model default transformer_tokenizer.model_max_length')
        if sentence_pair: self.max_seq_len=ifnone(max_seq_len, self.tok.max_len_sentences_pair) 
        else: self.max_seq_len=ifnone(max_seq_len, self.tok.max_len_single_sentence)
        self.pretrain_id = pretrain_id
        
    def do_tokenize(self, o):
        #print(o,flush=True)
        (o,o2) = o

        """Limits the maximum sequence length and add the special tokens"""
        CLS, SEP=self.tok.cls_token, self.tok.sep_token
        
        # Add prefix space, depending on model selected
        if 'roberta' in self.pretrain_id: tokens=self.tok.encode_plus(o, o2,add_prefix_space=True, return_token_type_ids= True, return_attention_mask= True)
        else: tokens = self.tok.encode_plus(o,o2, return_token_type_ids= True, return_attention_mask= True)
        
        return (tokens['input_ids'][:self.max_seq_len],tokens['attention_mask'][:self.max_seq_len],tokens['token_type_ids'][:self.max_seq_len])

        # order of 'tokens', 'SEP' and 'CLS'
        #if 'xlnet' in self.pretrain_id: return tokens + [SEP] +  [CLS]
        #else: return [CLS] + tokens + [SEP]

    def __call__(self, items): 
        #print(items,flush=True)
        #for o in items: 
        return self.do_tokenize(items)


def ds_pad_input(samples, pad_idx=1, pad_fields=0, pad_first=False, backwards=False):
    "Function that collect `samples` and adds padding"
    inputs_field = 0
    pad_fields = L(pad_fields)
    max_len_l = pad_fields.map(lambda f: max([len(s[inputs_field][f]) for s in samples]))
    if backwards: pad_first = not pad_first
    def _f(field_idx, inputs):
        if field_idx not in [inputs_field]: return inputs

        padded_inputs = []

        for x in inputs:

          idx = pad_fields.items.index(field_idx) #TODO: remove items if L.index is fixed

          sl = slice(-len(x), sys.maxsize) if pad_first else slice(0, len(x))
          pad =  x.new_zeros(max_len_l[idx]-x.shape[0])+pad_idx
          x1 = torch.cat([pad, x] if pad_first else [x, pad])
          if backwards: x1 = x1.flip(0)

          padded_inputs.append(retain_type(x1, x))
        return tuple(padded_inputs)

    return [tuple(map(lambda idxx: _f(*idxx), enumerate(s))) for s in samples]

def transformer_padding(tokenizer=None, max_seq_len=None, sentence_pair=False): 
    if tokenizer.padding_side == 'right': pad_first=False
    else: pad_first=True
    max_seq_len = ifnone(max_seq_len, tokenizer.model_max_length) 
    
    return partial(ds_pad_input, pad_first=pad_first, pad_idx=tokenizer.pad_token_id,pad_fields=[0,1,2])


class DSConfig(object):
  r""" Base class for all configuration classes."""
  
  def from_env(env):
    """ Legacy compatibility method """
    (df,test_df,bs,max_seq_len,sentence_pair,eps,lr,epochs,patience,pretraineds,pretrain_id,models,results,tokenizer_vocab_ls,tokenizer,fai_tokenizer,use_activ) = env
    return DSConfig(df=df,test_df=test_df,bs=bs,max_seq_len=max_seq_len,sentence_pair=sentence_pair,eps=eps,lr=lr,epochs=epochs,patience=patience,pretraineds=pretraineds,pretrain_id=pretrain_id,models=models,results=results,tokenizer_vocab_ls=tokenizer_vocab_ls,tokenizer=tokenizer,fai_tokenizer=fai_tokenizer,use_activ=use_activ)

  def __init__(self, **kwargs):

      # Attributes with defaults

      # Path ai test e training set
      self.df_path = kwargs.pop("df_path", None)
      self.test_df_path = kwargs.pop("test_df_path", None) 

      # Batch size
      self.bs = kwargs.pop("bs", 32) 

      # Limite sequenza (in base al preaddestramento)
      self.max_seq_len = kwargs.pop("max_seq_len", 512) 

      # Parametro deprecato
      self.sentence_pair = kwargs.pop("sentence_pair", False) 

      # Parametro Eps per ottimizzatore AdamW
      self.eps = kwargs.pop("eps", 0.0001) 

      # Learning rate
      self.lr = kwargs.pop("lr", 1e-5) 

      # Numero massimo di epoch
      self.epochs = kwargs.pop("epochs", 2000) 

      # Numero massimo di epoch per cui la validation loss può non migliorare, prima dello stop del training
      self.patience = kwargs.pop("patience", 20) 
      
      # Numero massimo di epoch per cui la validation loss può non migliorare, prima della riduzione del learning rate
      self.plateau_patience = kwargs.pop("plateau_patience", 5) 

      # Lista di pesi di pre-addestramento
      self.pretraineds = kwargs.pop("pretraineds", []) 

      # Indice dei pesi di pre-addestramento da usare per la run
      self.pretrain_id = kwargs.pop("pretrain_id", 0) 
      
      # Lista di modelli
      self.models = kwargs.pop("models", []) 

      # Contenitore delle metriche     
      self.results = kwargs.pop("results", {}) 
      
      # Numero di split per la cross validation
      self.n_splits = kwargs.pop("n_splits", 10) 


      # Utilizza funzione di attivazione ReLU prima dell'ultimo livello

      self.use_activ = kwargs.pop("use_activ", True) 

      # Indica sino a quale gruppo di layer congelare l'apprendimento
      self.freeze_to = kwargs.pop("freeze_to",  1)

      # Numero di filtri per la cnn
      self.kernel_num = kwargs.pop("kernel_num",  3)

      # Dimensione dei filtri della cnn
      self.kernel_sizes = kwargs.pop("kernel_sizes",  [2, 3, 4])

      # Numero di feature per parola
      self.embed_dim = kwargs.pop("embed_dim",  768)

      # Probabilità di dropout
      self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob",  0.1)

      # Indica se l'output è multi-etichetta
      self.multi_label = kwargs.pop("multi_label",  False) # multiple classifications and labels     

      # Numero di etichette (calcolato automaticamente, ma imponibile)
      self.num_labels = kwargs.pop("num_labels", None)

      # Utilizza come input sia la domanda che la risposta (altrimenti il dataloader usa solo la risposta)
      self.use_qa = kwargs.pop("use_qa", False)

      # Esegue 10-fold validation se True
      self.cross_validation = kwargs.pop("cross_validation",  False)

      # Attiva label smoothing (default a 0.9)
      self.label_smoothing = kwargs.pop("label_smoothing",  False)

      # Lista dei learner generati (viene riempita automaticamente durante l'addestramento)
      self.learner_datalist = kwargs.pop("learner_datalist",  [])

      # Flag (vengono manipolate automaticamente durante l'esperimento)
      self.started = kwargs.pop("started",  False)
      self.over = kwargs.pop("over",  False)

  def __setitem__(self, key, value):
    if self.started:
      while True:
        confirm = input("Experiment has already started.\nDo you really want to overwrite this param and start over? Y to confirm, any other to cancel")
        if (confirm == 'Y'):
          self.learner_datalist = []
          self.started = False
          self.over = False
          self.averages = None
          self.results = None
          self.__dict__[key] = value
    else:
      self.__dict__[key] = value

def ds_loop_gen(env):
	dsc = DSConfig.from_env(env)
	return partial(_ds_loop,dsc=dsc)

def _ds_loop(model_idx,dsc):
	
	dsc.model_idx = model_idx
	exp = DSExperiment(dsc)
	return exp.run() 


class DSExperiment(object):

  def __init__(self, dsc : DSConfig, *args, **kwargs):

    while True:
      self.name = input("Enter a name for this experiment: ")
      if len(self.name) > 0:
        break
        
    self.dsc = dsc

    self.save_models_path = '/content/drive/My Drive/dnlp_models/'
    self.save_exp_path = '/content/drive/My Drive/dnlp_exps/'

    self.filename = self.save_exp_path + self.name


    self.load()

    self.df = pd.read_csv(self.dsc.df_path, na_filter=False)
    self.test_df = pd.read_csv(self.dsc.test_df_path, na_filter=False)

    if (hasattr(dsc,'label_attr')):
      self.df['label'] = self.df[dsc.label_attr]
      self.test_df['label'] = self.test_df[dsc.label_attr]

    self.experiment_setup()

    self.print_config()


  def started(self):
    return len(self.dsc.learner_datalist) > 0

  def over(self):
    return ((not self.dsc.cross_validation) and self.started()) or (len(self.dsc.learner_datalist) == self.dsc.n_splits)

  def copy(self):

    dsc_copy = copy.deepcopy(self.dsc)
    dsc_copy.started = False
    dsc_copy.over = False
    dsc_copy.learner_datalist = []
    return DSExperiment(dsc_copy)

  def name_training(self):
    dsc = self.dsc

    # current date and time
    now = datetime.now()

    timestamp = now.isoformat(sep='_', timespec='seconds')
    
    training_id = dsc.pretrain_id + '/' + timestamp + ' ' + dsc.model_class_name + ' - lr: ' + str(dsc.lr)

    return training_id

  def create_learner(self, train_index, valid_index, training_id = None, is_test=False):

    dsc = self.dsc
    pretrained = False

    if is_test:
      df = self.test_df
    else:
      df = self.df
      df.loc[train_index,'is_valid'] = False
      df.loc[valid_index,'is_valid'] = True

    if training_id:
      pretrained = True
    else:
      training_id = self.name_training()

    fname = self.save_models_path + training_id

    splits = ColSplitter()(df)

    dsets = Datasets(df, splits=splits, tfms=self.tfms, dl_type=SortedDL)
    
    dls = dsets.dataloaders(bs=dsc.bs, before_batch=self.before_batch)

    assert (dsc.num_labels == dls.c)

    fai_model = dsc.model_cls(config_dict = dsc.tf_config, tokenizer = self.tokenizer, model_name = dsc.pretrain_id, pretrained = True, use_activ=dsc.use_activ)

    self.callbacks = [SaveModelCallback(fname=fname),EarlyStoppingCallback(patience=dsc.patience),ReduceLROnPlateau(patience=dsc.plateau_patience)]

    learn = Learner(dls, fai_model,  opt_func=self.opt_func, loss_func=self.criterion, metrics=self.metrics, cbs=self.callbacks, splitter=fai_model.transformer_spltr).to_fp16()

    if (pretrained):
     learn.load(fname)

    print(training_id, flush=true)
    return training_id, learn

  def transformer_setup(self):
    dsc = self.dsc
    tf_config = AutoConfig.from_pretrained(dsc.pretrain_id)

    # Number of classifier classes
    tf_config.num_labels = dsc.num_labels

    # Copy from DSConfig to BertConfig

    # Bert Params
    tf_config.hidden_dropout_prob = dsc.hidden_dropout_prob

    # CNN Params
    tf_config.embed_dim = dsc.embed_dim
    tf_config.embed_num = dsc.max_seq_len*4
    tf_config.kernel_num = dsc.kernel_num
    tf_config.kernel_sizes = dsc.kernel_sizes

    tf_config.max_seq_len = dsc.max_seq_len

    dsc.tf_config = tf_config    

  def experiment_setup(self):
    dsc = self.dsc

    dsc.num_labels = self.df['label'].nunique()

    self.tokenizer = AutoTokenizer.from_pretrained(dsc.pretrain_id)
    self.tokenizer_vocab=self.tokenizer.get_vocab() 

    self.tokenizer_vocab_ls = [k for k, v in sorted(self.tokenizer_vocab.items(), key=lambda item: item[1])]

    self.dsc.max_seq_len = min(dsc.max_seq_len,self.tokenizer.model_max_length)
      
    self.tok_func = TokFunc(transformer_tokenizer=self.tokenizer, pretrain_id=self.dsc.pretrain_id, max_seq_len=self.dsc.max_seq_len, sentence_pair=self.dsc.sentence_pair)
    
    self.qa_tok_func = QATokFunc(transformer_tokenizer=self.tokenizer, pretrain_id=self.dsc.pretrain_id, max_seq_len=self.dsc.max_seq_len, sentence_pair=self.dsc.sentence_pair)
    
    self.fai_tokenizer = Tokenizer.from_df(text_cols='text', res_col_name='text', tok=self.tok_func, rules=[])

    model = dsc.models[dsc.model_idx]
    (dsc.model_cls, _) = model
    dsc.model_class_name = dsc.model_cls.__name__

    #outputstream = IPython.utils.io.Tee("experiment" + ".html", "a", channel="stdout")
    self.transformer_setup()


    print("\n\nWeights: %s\nModel: %s" % (dsc.pretrain_id, dsc.model_class_name), flush= True)
    ####
    if (not (dsc.started)):
      if (dsc.cross_validation):
        skf = StratifiedKFold(n_splits=dsc.n_splits, shuffle=True, random_state=42)
        X = self.df
        y = self.df['Rep']

        dsc.split_series = list(skf.split(X, y))

      else: # not cross validation
        dsc.split_series = [tuple(ColSplitter()(self.df))]
    
    #dsc.useRocAuc = (dsc.num_labels <= 3)
    dsc.useRocAuc = True
    #prova
    tfms = []

    if (dsc.use_qa):
      x_tfms = [DSTransform(self)]

    else:
      x_tfms = [attrgetter("text"), self.fai_tokenizer, Numericalize(vocab=self.tokenizer_vocab_ls)]
    
    tfms.append(x_tfms)
    
    if (0):
      x2_tfms = [attrgetter("Question"), self.fai_tokenizer, Numericalize(vocab=self.tokenizer_vocab_ls)]
      tfms.append(x2_tfms)

    if (dsc.multi_label):
      y_tfms = [attrgetter("labels"), MultiCategorize()]
      criterion = BCEWithLogitsLossFlat()
      metrics = [accuracy_multi,F1ScoreMulti(average='macro')]

    else:
      labels = self.df['label'].unique().tolist()
      labels.sort()
      y_tfms = [attrgetter("label"), Categorize(vocab=labels)]
      if (dsc.label_smoothing):
        criterion = WeightedLabelSmoothingCrossEntropy(dsc,self.df)
      else:
        criterion = WeightedCrossEntropyLoss(dsc,self.df)
    
      metrics = [accuracy,FBeta(1,average='macro'), MatthewsCorrCoef()]
    
      if (dsc.useRocAuc):
        metrics.append(RocAuc(multi_class='ovo'))
    

    tfms.append(y_tfms)      

    self.tfms = tfms
    self.criterion = criterion  
    self.metrics = metrics

    self.opt_func = partial(Adam, decouple_wd=True, eps=dsc.eps)

    self.before_batch=[transformer_padding(self.tokenizer)]

  def print_config(self, config = None):
    #ignore_attrs = ['tf_config', 'tokenizer_vocab_ls', 'tokenizer', 'fai_tokenizer', 'tokenizer_vocab', 'tfms', 'split_series', 'before_batch', 'callbacks', 'tok_func', 'qatok_func','models','pretraineds']
    ignore_attrs = ['tf_config']

    print("Exp name: %s" % (self.name,))
    print("Training conf:")

    dsc = config if config else self.dsc

    for attr, value in vars(dsc).items():
      if (attr in ['df', 'test_df']):
        print(('{}: {}\n[{}]').format(attr,len(value),value.columns),flush=True)
      elif (not(attr in ignore_attrs)):
        print(('{}: {}').format(attr,value),flush=True)
      else:
        print(attr,flush=True)

  def load(self):

      try:
        f = open(self.filename, 'rb')
        tmp_dict = pickle.load(f)

        f.close()          

        self.dsc.__dict__.update(tmp_dict) 

        print('Resuming experiment named: %s' % (self.name,))

      except FileNotFoundError:
        pass

  def save(self):
      f = open(self.filename, 'wb')
      pickle.dump(self.dsc.__dict__, f, 2)
      f.close()


  def run(self):

    dsc = self.dsc

    if (not dsc.over):

      start_from_split = len(dsc.learner_datalist) if dsc.started else 0

      if (start_from_split > 0):
        print('Resuming from split: %s' % (start_from_split,))

      for train_index, valid_index in dsc.split_series[start_from_split:]:
      	training_id, learn = self.create_learner(train_index, valid_index)
        
      	if(('modello.json').exists()):



        	json_file = open('modello.json', 'r')
        	loaded_model_json = json_file.read()
        	json_file.close()
        	learn = model_from_json(loaded_model_json)

    
        		


        if (dsc.epochs > 0):
          learn.freeze_to(dsc.freeze_to)

          #learn.lr_find()
          plt.show()
        
          try:
            #with learn.no_mbar(): learn.fit_one_cycle(epochs, lr_max=lr)
            with learn.no_mbar(): learn.fit(dsc.epochs, lr=dsc.lr, wd=1e-4, reset_opt=False)
          except KeyboardInterrupt:
            print('Fit was interrupted')
            self.save()
        
          learn.recorder.plot_loss(skip_start=0)
          plt.show()

        dsc.learner_datalist.append((training_id, train_index, valid_index))
        learn_json=model.to_json()
        with open("modello.json", "w") as json_file:
        	json_file.write(model_json)
        del learn
        gc.collect()
        torch.cuda.empty_cache()
        
        dsc.started = self.started()
        dsc.over = self.over()
        self.save()

    if (dsc.epochs > 0):
      self.benchmark()

    #outputstream.close()

 
  def benchmark(self):
    
    dsc = self.dsc

    if (not hasattr(self.dsc, 'averages')):

      metricsL = ["loss", "accuracy", "fbeta", "mcc", "rocauc"]

      benchmarks = ['valid','test']

      results = { benchmark: {metric: [] for metric in metricsL} for benchmark in benchmarks }
      averages = { benchmark: {metric: None for metric in metricsL} for benchmark in benchmarks }
      confusion_matrices = { benchmark: [] for benchmark in benchmarks }
      classification_reports = { benchmark: [] for benchmark in benchmarks }
      
      
      

      
      learner_datalist = self.dsc.learner_datalist



      for (i, learner_data) in enumerate(learner_datalist, start = 1):



        training_id, train_index, valid_index = learner_data

        for benchmark in benchmarks:

          training_id, learner = self.create_learner(train_index, valid_index, training_id, 'test' == benchmark)

          print('*'*30)
          print('%s metrics for %s' % (benchmark, training_id))

          metrics = learner.validate()
          
          for (metric, value) in zip(metricsL, metrics):
            results[benchmark][metric].append(value)
            print(results)

            interp = DSClassificationInterpretation.from_learner(learner)

          confusion_matrices[benchmark].append(interp.confusion_matrix())
          classification_reports[benchmark].append(interp.classification_report())

          if ('valid' == benchmark):

            figsize = figsize=(10, 10) if dsc.num_labels > 3 else None

            interp.plot_confusion_matrix(figsize=figsize)
            plt.show()

            interp.print_classification_report()

          del learner
          gc.collect()    
          torch.cuda.empty_cache()

          if (i == len(learner_datalist)):
            averages[benchmark] = { metric: mean(results[benchmark][metric]) if len(results[benchmark][metric]) > 0 else None for metric in metricsL }

      self.dsc.results = results
      self.dsc.averages = averages
      self.dsc.confusion_matrices = confusion_matrices
      self.dsc.classification_reports = classification_reports
      print('\n'*2)
      self.save()

    interp.plot_average_confusion_matrix(self.dsc.confusion_matrices['test'],figsize=(12,12))
    interp.print_average_classification_report(self.dsc.classification_reports['test'])
    interp.auc()
    pprint(self.dsc.averages)



class DSClassificationInterpretation(ClassificationInterpretation):
    "Interpretation methods for DS classification models."

    def __init__(self, dl, inputs, preds, targs, decoded, losses):
        super().__init__(dl, inputs, preds, targs, decoded, losses)


    def print_average_classification_report(self,classification_reports):
        report_list = list()
        for report in classification_reports:
            splited = [' '.join(x.split()) for x in report.split('\n')]
        
            header = [x for x in splited[0].split(' ')]
        
            data_by_label = splited[2:-5]
            accuracy_data = splited[-4]
            macro_avg_data = splited[-3]
            weighted_avg_data = splited[-2]
            
            data = np.array([element for line in data_by_label for element in line.split(' ')] )

            labels = np.array([label for label in data[::5]] )

            data = data.reshape(-1, len(header) + 1)


            data = np.delete(data, 0, 1).astype(float)

            accuracy = np.array([0] + [0] + accuracy_data.split(' ')[-2:]).astype(float).reshape(-1, len(header))
            macro_avg = np.array(macro_avg_data.split(' ')[-4:]).astype(float).reshape(-1, len(header))
            weighted_avg = np.array(weighted_avg_data.split(' ')[-4:]).astype(float).reshape(-1, len(header))
            
            df = pd.DataFrame(np.concatenate((data, accuracy, macro_avg,weighted_avg)), columns=header)
             
            report_list.append(df)
        res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)

        rename_index = { i: label for (i,label) in enumerate(labels) }
        
        rename_index.update({ len(rename_index) : 'accuracy' })
        rename_index.update({ len(rename_index) : 'macro avg' })
        rename_index.update({ len(rename_index) : 'weighted avg' })
        
        res['precision'] = res['precision'].apply(partial(round,ndigits=2))
        res['recall'] = res['recall'].apply(partial(round,ndigits=2))
        res['support'] = res['support'].apply(int)
        
        res.iloc[-3,[0,1]] = ' '      
        report_average = res.rename(index=rename_index)        

        display(HTML(report_average.to_html(float_format=lambda x: '%.2f' % x)))
    
    def auc(self):
        "Print scikit-learn classification report"
        d,t = flatten_check(self.decoded, self.targs)
        #clf = sk(solver="liblinear").fit(d, t)
  



    def classification_report(self):
        "Print scikit-learn classification report"
        d,t = flatten_check(self.decoded, self.targs)
        return skm.classification_report(t, d, labels=list(self.vocab.o2i.values()), target_names=[str(v) for v in self.vocab])

    def plot_average_confusion_matrix(self, confusion_matrices, title='Confusion matrix', normalized=False, cmap="Reds", plot_txt=True, **kwargs):
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs

        normalized_matrices = []
        support = confusion_matrices[0].sum(axis=1)

        # Normalize matrices
        for m in confusion_matrices:
          nm = m.astype('float') / m.sum(axis=1)[:, np.newaxis] 
          normalized_matrices.append(nm)

        ncm = np.mean(normalized_matrices,axis=0)

        fig = plt.figure(**kwargs)
        plt.imshow(ncm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(self.vocab))
        plt.xticks(tick_marks, self.vocab, rotation=90)
        plt.yticks(tick_marks, self.vocab, rotation=0)

        if (normalized):
          cm = ncm
          norm_dec = 2
          
        else:
          cm = np.mean(confusion_matrices,axis=0)
          norm_dec = 0

        if plot_txt:
            thresh = ncm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]*100:.{0}f}%' if normalized else f'{cm[i, j]:.{0}f}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if ncm[i, j] > thresh else "black")

        ax = fig.gca()
    
        ax2 = ax.twinx()
        ax.set_ylim(len(self.vocab)-.5,-.5)
        ax2.set_ylim(len(self.vocab)-.45,-.55)
  
        plt.tight_layout()
        
        ax2.set_yticks(tick_marks)
        ax2.set_yticklabels(support, {'horizontalalignment':"left", 'verticalalignment':"center"})
        ax2.tick_params(pad=10,rotation=0, )        


        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)    