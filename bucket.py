#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
from configurable import Configurable

#***************************************************************
class Bucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(Bucket, self).__init__(*args, **kwargs)
    self._size = None
    self._data = None
    self._sents = None
    
    return
  
  #=============================================================
  def reset(self, size):
    """"""
    
    self._size = size
    self._data = []
    self._sents = []
    self._polo = []
    return
  
  #=============================================================
  def add(self, sent):
    """"""
    
    if isinstance(self._data, np.ndarray):
      raise TypeError("The buckets have already been finalized, you can't add more")
    if len(sent) > self.size and self.size != -1:
      raise ValueError('Bucket of size %d received sequence of len %d' % (self.size, len(sent)))
    words = [word[0] for word in sent]# remove root
    idxs = [word[1:] for word in sent]
    sentpolo = max([word[6] for word in sent])
    num = min([word[1] for word in sent])
    assert num > 0,'Error in WordID'
    if sentpolo == 0:
      sentpolo = [1, 0, 0] #negative
    elif sentpolo == 1:
      sentpolo = [0, 1, 0] #neutral
    elif sentpolo == 2:
      sentpolo = [0, 0, 1] #positive
    else:
      raise ValueError('Too many polority to pack') 
#    print(sentpolo)
#    print(len(sent[0]))
#    print(sent[0])
#    sys.exit(0)
    self._sents.append(words)
    self._data.append(idxs)
    self._polo.append(sentpolo)
    return len(self._data)-1
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self._data is None:
      raise ValueError('You need to reset the Buckets before finalizing them')
    
    if len(self._data) > 0:
      shape = (len(self._data), self.size, len(self._data[-1][-1]))
      data = np.zeros(shape, dtype=np.int32)
      for i, datum in enumerate(self._data):
        datum = np.array(datum)
        data[i, 0:len(datum)] = datum
      self._data = data
      self._sents = np.array(self._sents)
      self._polo = np.array(self._polo)
    else:
      self._data = np.zeros((0,1), dtype=np.float32)
      self._sents = np.zeros((0,1), dtype=str)
      self._polo = np.zeros((0,1), dtype=np.float32)
#    print(self._data[1999])
#    print(self.polo[1999])
#    sys.exit(0)
    print('Bucket %s is %d x %d' % ((self._name,) + self._data.shape[0:2]))
    return
  
  #=============================================================
  def __len__(self):
    return len(self._data)
  
  #=============================================================
  @property
  def size(self):
    return self._size
  @property
  def data(self):
    return self._data
  @property
  def sents(self):
    return self._sents
  @property
  def polo(self):
    return self._polo
