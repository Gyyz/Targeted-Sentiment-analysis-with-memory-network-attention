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
class MCAttention(BaseAttentions):
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
                top_recur = tf.nn.dropout(top_recur, 0.6, seed=666)  # batch_size, bucket_size, lstm_dim(300)
            # =======================================================
            htscore = self.getTarHd(top_recur, istarget)  # get the target_word average score

            attenrseq = attenlseq = attenwseq = htscore

            for j in range(self.n_remem):
                with tf.variable_scope('MEMWAtt%d'%j, reuse=reuse):
                    input_size = attenwseq.get_shape().as_list()[-1]
                    batch_size = tf.shape(attenwseq)[0]
                    input_shape = tf.pack([batch_size, 1, input_size])
                    out_shape = tf.pack([batch_size, input_size])
                    attenwseq = tf.reshape(attenwseq, input_shape)
                    attenwseq,_ = self.MRNN(attenwseq)
                    attenwseq = tf.reshape(attenwseq, out_shape)
                    attenwseq = self.compAtt(top_recur, attenwseq, nontarget, scope='allatt')

            for j in range(self.n_remem):
                with tf.variable_scope('MEMLAtt%d'%j, reuse=reuse):
                    input_size = attenlseq.get_shape().as_list()[-1]
                    batch_size = tf.shape(attenlseq)[0]
                    input_shape = tf.pack([batch_size, 1, input_size])
                    out_shape = tf.pack([batch_size, input_size])
                    attenlseq = tf.reshape(attenlseq, input_shape)
                    attenlseq,_ = self.MRNN(attenlseq)
                    attenlseq = tf.reshape(attenlseq, out_shape)
                    attenlseq = self.compAtt(top_recur, attenlseq, nontarget, scope='leftatt')

            for j in range(self.n_remem):
                with tf.variable_scope('MEMRAtt%d'%j, reuse=reuse):
                    input_size = attenrseq.get_shape().as_list()[-1]
                    batch_size = tf.shape(attenrseq)[0]
                    input_shape = tf.pack([batch_size, 1, input_size])
                    out_shape = tf.pack([batch_size, input_size])
                    attenrseq = tf.reshape(attenrseq, input_shape)
                    attenrseq,_ = self.MRNN(attenrseq)
                    attenrseq = tf.reshape(attenrseq, out_shape)
                    attenrseq = self.compAtt(top_recur, attenrseq, nontarget, scope='rightatt')

            sntVec = self.Seq2Pb(attenwseq, attenlseq, attenrseq)
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
