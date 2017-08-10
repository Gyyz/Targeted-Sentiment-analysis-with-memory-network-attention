#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
from vocab import Vocab
from lib.models.sentiment.baseatt import BaseAttentions


# ***************************************************************
class CAttention(BaseAttentions):
    """"""

    # =============================================================
    def __call__(self, dataset, moving_params=None):
        """"""

        vocabs = dataset.vocabs
        inputs = dataset.inputs
        targets = dataset.targets
        reuse = (moving_params is not None)
        # self.reuse = reuse
        self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:, :, 0], vocabs[0].ROOT)), 2)
        self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1, 1])
        self.n_tokens = tf.reduce_sum(self.sequence_lengths)
        self.moving_params = moving_params

        word_inputs = vocabs[0].embedding_lookup(inputs[:, :, 0], inputs[:, :, 1], moving_params=self.moving_params)
        istarget = inputs[:, :, 2]  # batch_size, bucket_size
        bftarget = inputs[:, :, 3]  # same above
        aftarget = inputs[:, :, 4]  # same above
        nontarget = 1 - istarget
        top_recur = self.embed_concat(word_inputs)  # batch_size, bucket_size, word_dim(200)
        # Bottom/Original RNN layers
        for i in xrange(self.n_recur):
            with tf.variable_scope('RNN%d' % i, reuse=reuse):
                top_recur, _ = self.RNN(top_recur)
                # ====================================================================================================
        with tf.variable_scope('Attention_context', reuse=reuse):
            """"""
            if moving_params is None:
                top_recur = tf.nn.dropout(top_recur, 0.6, seed=76)  # batch_size, bucket_size, lstm_dim(300)
            # =======================================================
            htscore = self.getTarHd(top_recur, istarget)  # get the target_word average score

            attenlseq = self.compAtt(top_recur, htscore, bftarget, scope='leftatt')  # get s'= sum(alpha*hi) ==> h_s
            attenrseq = self.compAtt(top_recur, htscore, aftarget, scope='rightatt')
            attenwseq = self.compAtt(top_recur, htscore, nontarget, scope='allatt')
            if moving_params is None:
                attenrseq = tf.nn.dropout(attenrseq, 0.5, seed=77)
                attenwseq = tf.nn.dropout(attenwseq, 0.5, seed=77)
                attenlseq = tf.nn.dropout(attenlseq, 0.5, seed=77)

            sntVec = self.Seq2Pb(attenwseq, attenlseq, attenrseq, reuse=reuse)
            attout = self.attoutput(sntVec, targets)
            return attout
            # ======================================================================================================================
            #
            # #=============================================================

    def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
        """"""

        parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
        rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
        rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
        return parse_preds, rel_preds
