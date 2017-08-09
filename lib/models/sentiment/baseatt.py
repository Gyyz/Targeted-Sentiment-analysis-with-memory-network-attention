#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from collections import Counter
from vocab import Vocab
from lib.models import NN


# ***************************************************************
class BaseAttentions(NN):
  """"""

  # =============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    raise NotImplementedError

  # =============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    raise NotImplementedError

  #==============================================================
  def getTarHd(self, word_hlstm, targetwd_idxs, comput_idxs=None):
    """
    :param word_hlstm: the hidden lstm output
    :param comput_idxs: the id information of the word t
    :param targetwd_idxs: the id information of target wordo compute
    :return: the attention lstm concat with target attention( whole snt/ before target words snt/ after target words snt)
    """
    # Get shape
    batch_size = tf.shape(word_hlstm)[0]
    bucket_size = tf.shape(word_hlstm)[1]
    lstm_dim = word_hlstm.get_shape().as_list()[-1]
    input_shape = tf.pack([batch_size, bucket_size, lstm_dim])
    target_wd_shape = tf.pack([batch_size, bucket_size, 1])#batch_size, bucket_size, 1
    lstm_shapes = tf.ones(input_shape) # batch_size, bucket_size, lstm_dim
    bt_shape = tf.pack([batch_size,1])
    
    # Get attention matrix
    raw_score = tf.mul(word_hlstm, tf.reshape(tf.to_float(targetwd_idxs), target_wd_shape)) #[M, N, P] X [M, N, 1] ==> [M, N, 1] only target words hidden are reserved
    word_hlstm_sum = tf.reduce_sum(raw_score, axis=1) # sum_up of target hidden state
    ht_num = tf.to_float(tf.reduce_sum(targetwd_idxs, axis=1)) # target_word number [batch_size]
#    ht_score = tf.transpose(word_hlstm_sum) / ht_num # average hidden state of target words [M,P]/[M] ==> [P, M] / [M]
#    ht_score = tf.transpose(ht_score) # why transpose? ==> 
    ht_score = word_hlstm_sum / tf.reshape(ht_num, bt_shape) # [batch_size, lstm_dim*2

    return ht_score

  #=================================================================
  def compAtt(self, wd_lstm, ht_avhd, attenpart, scope = 'alphaseq', reuse=None):

    batch_size = tf.shape(wd_lstm)[0]
    bucket_size = tf.shape(wd_lstm)[1]
    input_size = wd_lstm.get_shape().as_list()[-1]
    input_shape = tf.pack([batch_size, bucket_size, input_size])
    attshape = tf.pack([batch_size, bucket_size, 1]) #outputshape
    lstm_shape = tf.ones(input_shape, dtype=tf.float32)

    attenpart = tf.mul(lstm_shape, tf.reshape(tf.to_float(attenpart), attshape)) #[M, N, P] X [M, N, 1] ==> [M, N, PX1]
    attenlstm = tf.mul(wd_lstm, attenpart)

    htlstm = tf.mul(tf.transpose(attenpart, (1, 0, 2)), ht_avhd) #[M, N, P] X [P]
    htlstm = tf.transpose(htlstm, (1, 0, 2))

    attenhd = tf.concat(2, [attenlstm, htlstm]) #[h_i , h_t]

    att_hd_dim = attenhd.get_shape().as_list()[-1]
    output_shape = tf.pack([batch_size, bucket_size, 1])
    with tf.variable_scope(scope, reuse=reuse):
      weight_almat = tf.get_variable('alpha_matrix', [att_hd_dim, 1], initializer=tf.random_normal_initializer())
      bias_almat = tf.get_variable('att_bias', [1], initializer=tf.zeros_initializer)
      att_lstm_rp = tf.reshape(attenhd, [-1, att_hd_dim])
      raw_belta = tf.matmul(att_lstm_rp, weight_almat) + bias_almat
      wd_belta = tf.tanh(raw_belta)

      # Get alpha and the sentence representation
      wd_alpha = tf.reshape(wd_belta, output_shape)
      wd_salpha = tf.nn.softmax(wd_alpha, 1)
      AlphaSeq = tf.mul(wd_lstm, wd_salpha)
      AlphaRpt = tf.reduce_sum(AlphaSeq, 1)

    return AlphaRpt


  # compute the alpha sentence representation
  def Seq2Pb(self, alpha_seqt, left_seqt=None, right_seqt=None, relu=False, hitarget=None, output_size=3, reuse=None):
    """
    :param alpha_seqt: the whole sentence alpha representation
    :return: the probablity vector of sentiment
    """
    if left_seqt is None and right_seqt is None:
      with tf.variable_scope('Base', reuse=reuse):
        # Get the input shape
        ndims = len(alpha_seqt.get_shape().as_list())
        input_shape = tf.shape(alpha_seqt)
        input_size = alpha_seqt.get_shape().as_list()[-1]
        output_shape = []
        batch_size = input_shape[0]
        output_shape.append(input_shape[0])
        output_shape.append(output_size)
        output_shape = tf.pack(output_shape)
        alpha_seqt = tf.reshape(alpha_seqt, tf.pack([batch_size, input_size]))

        # Get the Matrix and Multiply
        weight_matrix = tf.get_variable('AttSent', [input_size, output_size],
                                            initializer=tf.random_normal_initializer())
        bias_matrix = tf.get_variable('AttSentBias', [3], initializer=tf.zeros_initializer)
        SentiVector = tf.matmul(alpha_seqt, weight_matrix) + bias_matrix

        return SentiVector

    if left_seqt is not None and right_seqt is not None:
      with tf.variable_scope('Context', reuse=reuse):
        linput_size = left_seqt.get_shape().as_list()[-1]
        rinput_size = right_seqt.get_shape().as_list()[-1]
        winput_size = alpha_seqt.get_shape().as_list()[-1]
        batch_size = tf.shape(alpha_seqt)[0]

        left_seqt = tf.reshape(left_seqt, tf.pack([batch_size, linput_size]))
        right_seqt = tf.reshape(right_seqt, tf.pack([batch_size, rinput_size]))
        whole_seqt = tf.reshape(alpha_seqt, tf.pack([batch_size, winput_size]))

        lweight = tf.get_variable('Leftcon', [linput_size, output_size], initializer=tf.random_normal_initializer())
        rweight = tf.get_variable('Rightcon', [rinput_size, output_size], initializer=tf.random_normal_initializer())
        wweight = tf.get_variable('Wholecon', [winput_size, output_size], initializer=tf.random_normal_initializer())
        bias_mat = tf.get_variable('ContextBias', [3], initializer=tf.zeros_initializer)

        SentiVector = tf.matmul(left_seqt, lweight) + tf.matmul(right_seqt, rweight) + tf.matmul( whole_seqt, wweight) + bias_mat

        return SentiVector

    if relu is True and hitarget is not None:
      with tf.variable_scope('GateContext', reuse=reuse):
        linput_size = left_seqt.get_shape().as_list()[-1]
        rinput_size = right_seqt.get_shape().as_list()[-1]
        winput_size = alpha_seqt.get_shape().as_list()[-1]
        ht_size = hitarget.get_shape().as_list()[-1]
        batch_size = tf.shape(alpha_seqt)[0]

        left_seqt = tf.reshape(left_seqt, tf.pack([batch_size, linput_size]))
        right_seqt = tf.reshape(right_seqt, tf.pack([batch_size, rinput_size]))
        whole_seqt = tf.reshape(alpha_seqt, tf.pack([batch_size, winput_size]))

        lweight = tf.get_variable('Left', [linput_size, ht_size], initializer=tf.random_normal_initializer())
        rweight = tf.get_variable('Right', [rinput_size, ht_size], initializer=tf.random_normal_initializer())
        wweight = tf.get_variable('Whole', [winput_size, ht_size], initializer=tf.random_normal_initializer())

        ltweight = tf.get_variable('LeftTar', [ht_size, ht_size], initializer=tf.random_normal_initializer())
        rtweight = tf.get_variable('RightTar', [ht_size, ht_size], initializer=tf.random_normal_initializer())
        wtweight = tf.get_variable('WholeTar', [ht_size, ht_size], initializer=tf.random_normal_initializer())

        bias_matl = tf.get_variable('LConBias', [ht_size], initializer=tf.zeros_initializer)
        bias_matr = tf.get_variable('RConBias', [ht_size], initializer=tf.zeros_initializer)
        bias_matw = tf.get_variable('WConBias', [ht_size], initializer=tf.zeros_initializer)

        zl = tf.matmul(left_seqt, lweight) + tf.matmul(hitarget, ltweight) + bias_matl
        zr = tf.matmul(right_seqt, rweight) + tf.matmul(hitarget, rtweight) + bias_matr
        zw = tf.matmul(whole_seqt, wweight) + tf.matmul(hitarget, wtweight) + bias_matw

        zsum = zl+zr+zw

        zlgate = zl/zsum
        zrgate = zr/zsum
        zwgate = zw/zsum

        zscore = tf.matmul(left_seqt, zlgate) + tf.matmul(right_seqt, zrgate) + tf.matmul(whole_seqt, zwgate)

        zweight = tf.get_variable('GATE', [ht_size, output_size], initializer=tf.random_normal_initializer())
        zbias = tf.get_variable('WConBi', [output_size], initializer=tf.zeros_initializer)

        SentiVector = tf.matmul(zscore, zweight) + zbias

        return SentiVector


  #=========================================================================================
  def attoutput(self, predlogits2D, glodenvec2D):
    """
    :param predlogits2D: the model output logits
    :param glodenvec2D:  target sentiment vec
    :return: the output dict
    """
    original_shape = tf.shape(predlogits2D)
    sent_num = original_shape[0]
    probabilities_2D = tf.nn.softmax(predlogits2D)

    logits1D = tf.to_int32(tf.argmax(predlogits2D, 1))
    targets1D = tf.to_int32(tf.argmax(glodenvec2D, 1))
    correct1D = tf.to_int32(tf.equal(logits1D, targets1D))
    n_correct = tf.reduce_sum(correct1D)
    accuracy = n_correct / sent_num

    cross_entropy1D = tf.nn.softmax_cross_entropy_with_logits(predlogits2D, glodenvec2D)
    loss = tf.reduce_mean(cross_entropy1D)

    
#    predictCounter = Counter(np.argmax(predlogits2D))
#    goldenCounter = Counter(np.argmax(glodenvec2D))

#    posCP = posCT = 0
#    negCP = negCT = 0
#    neuCP = neuCT = 0

#    posCP, negCP, neuCP = predictCounter.values()
#    posCT, negCT, neuCT = goldenCounter.values()

#    posCorrect = negCorrect = neuCorrect = 0

#    for pred, targ in zip(logits1D, targets1D):
#      if pred == 0 and targ == 0:
#        posCorrect += 1
#      if pred == 1 and targ == 1:
#        negCorrect += 1
#      if pred == 2 and targ == 2:
#        neuCorrect += 1
    
#    posF1 = self.getF1(posCorrect, posCP, posCT)
#    negF1 = self.getF1(negCorrect, negCP, negCT)
#    neuF1 = self.getF1(neuCorrect, neuCP, neuCT)
    
    attoutput = {
        'probabilities': probabilities_2D,
        'predictions': logits1D,
        'batch_size': sent_num,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'loss': loss}#,
#        'Fvalues': (posF1, negF1, neuF1)}
    return attoutput
  #==============================================================
  def getF1(self, correct, predict, golden):
    """
    :param correct: 
    :param predict: 
    :param golden: 
    :return: 
    """
    if correct == 0:
      return 0
    P = correct / float(predict)
    R = correct / float(golden)
    return P * R * 2 / (P + R)

  #=========================================================================================
  def validate(self, dt_inputs, dt_targets, dt_probs):
    """
    :param dt_input: the model input
    :param dt_targets:  target Gold
    :param dt_probs: targetPredict
    :return: the output dict
    """
    sents = []
    dt_targets = np.argmax(dt_targets, 1)
    dt_probs = np.argmax(dt_probs, 1)
    for inputs, targets, probs in zip(dt_inputs, dt_targets, dt_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      sent = -np.ones((length, 8), dtype=int)
      tokens = np.arange(0, length)
      sent[:,0:5] = inputs[tokens]
      sent[:,6] = targets
      sent[:,7] = probs
    return sents

  #========================================================================================
   


  @property
  def input_idxs(self):
    return (0, 1, 2, 3, 4)
  @property
  def target_idxs(self):
    return (5)

