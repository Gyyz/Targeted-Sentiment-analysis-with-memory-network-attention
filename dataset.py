#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
import sys
from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket

#***************************************************************
class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, filename, vocabs, builder, *args, **kwargs):
    """"""
    
    super(Dataset, self).__init__(*args, **kwargs)
    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.vocabs = vocabs
    self.rebucket()
    
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,3), name='targets')
    self.builder = builder()
  
  #=============================================================
  def file_iterator(self, filename):
    """"""
    
    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line:
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if buff[-1]:
                  buff.append([])
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line:
            buff[-1].append(line)
          else:
            if buff[-1]:
              buff.append([])
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff
  
  #=============================================================
  def _process_buff(self, buff):
    """"""
    
    words = self.vocabs[0]
    for i, sent in enumerate(buff):
      targetflag = 0
      for j, token in enumerate(sent):
	if token[1] != 'o':
	  targetflag = 1
#	print(token)
	word, istarget, bftarget, aftarget, sentmod = token[0], 0 if token[1] == 'o' else 1, 1 if token[1] =='o' and targetflag==0 else 0, 1 if token[1] == 'o' and targetflag == 1 else 0, self.getmood(token[1])
#	if istarget:
#          print(sentmod)
	buff[i][j] = (word,) + words[word] + (int(istarget),) + (int(bftarget),) + (int(aftarget),) + (sentmod,)
    return buff
  
  #=============================================================
  def getmood(self, polority):
    """"""
    if polority == 'o':
      return 0
    else:
      polority = polority.split('-')[1]
      if polority == 'positive':
        return 2
      elif polority == 'negative':
        return 0
      else:
	return 1
  #=============================================================
  def reset(self, sizes):
    """"""
    
    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return
  
  #=============================================================
  def rebucket(self):
    """"""
    
    buff = self._file_iterator.next()
    len_cntr = Counter()
    
    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)
    
    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return
  
  #=============================================================
  def _finalize(self):
    """"""
    
    self._metabucket._finalize()
    return
  
  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""
    
    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
#      print(len(bucket))
#      print(bucket.size)
      if batch_size == 0:
        n_splits = 1
      else:
        n_sent = len(bucket)
        n_splits = max(n_sent // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]
      polos = self[bkt_idx].polo[bkt_mb]
      assert len(data) == len(polos), 'MINI BATCH Errorrrrrrrrr'
#      print(polos.shape)
#      print(data.shape)
#      print(batch_size)
#      print(len(minibatches))
#      sys.exit(0)
      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))
      feed_dict = {
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: polos
      }
      yield feed_dict, sents
  
  #=============================================================
  def get_minibatches2(self, batch_size, input_idxs, target_idxs):
    """"""
    
    bkt_lens = np.empty(len(self._metabucket))
    for i, bucket in enumerate(self._metabucket):
      bkt_lens[i] = len(bucket)
    
    total_sents = np.sum(bkt_lens)
    bkt_probs = bkt_lens / total_sents
    n_sents = 0
    while n_sents < total_sents:
      n_sents += batch_size
      bkt = np.random.choice(self._metabucket._buckets, p=bkt_probs)
      data = bkt.data[np.random.randint(len(bkt), size=batch_size)]
      if bkt.size > 100:
        for data_ in np.array_split(data, 2):
          feed_dict = {
            self.inputs: data_[:,:,input_idxs],
            self.targets: data_[:,:,target_idxs]
          }
          yield feed_dict
      else:
        feed_dict = {
          self.inputs: data[:,:,input_idxs],
          self.targets: data[:,:,target_idxs]
        }
        yield feed_dict
  
  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts
  
  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
