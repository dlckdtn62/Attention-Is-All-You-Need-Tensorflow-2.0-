#!/usr/bin/env python
# coding: utf-8

# In[64]:


import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
import random


# In[65]:


EPOCHS = 200
NUM_WORDS = 2000


# In[99]:


class Scaled_Dot_Attention(tf.keras.Model):
    def __init__(self, d_emb, d_reduced, masked = False):
        super(Scaled_Dot_Attention, self).__init__()
        self.q = tf.keras.layers.Dense(d_reduced)
        self.k = tf.keras.layers.Dense(d_reduced)
        self.v = tf.keras.layers.Dense(d_reduced)
        ##차원 축소용
        self.scale = tf.keras.layers.Lambda(lambda x : x/np.sqrt(d_reduced))
        self.masked = masked
    def call(self, x, training = None, mask = None):
        q = self.scale(self.q(x[0]))
        k = self.k(x[1])
        k = tf.transpose(k, perm = [0, 2, 1])
        qk = tf.matmul(q, k)
        v = self.v(x[2])
        
        if self.masked == True:
            length = tf.shape(qk)[-1]
            mask = tf.fill((length, length), -np.inf)
            mask = tf.linalg.band_part(mask, 0, -1)
            ##더하는 개념에 있어서는 upper가 되어야 된다
            mask = tf.linalg.set_diag(mask, tf.zeros((length)))
            ##대각선에 대해서 행렬로 주어야 한다
            qk+=mask
        
        qk = tf.keras.layers.Activation('softmax')(qk)
        
        return tf.matmul(qk, v)


# In[100]:


class Multi_Head_Attention(tf.keras.Model):
    def __init__(self, h, d_emb, d_reduced, masked = False):
        super(Multi_Head_Attention, self).__init__()
        self.sequence = list()
        for _ in range(h):
            self.sequence.append(Scaled_Dot_Attention(d_emb, d_reduced, masked))
        self.dense = tf.keras.layers.Dense(d_emb)
        
    def call(self, x, training = False, mask = False):
        result = [layer(x, training, mask) for layer in self.sequence]
        result = tf.concat(result, axis = -1)
        return self.dense(result)


# In[101]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, h, d_reduced):
        super(Encoder, self).__init__()
        self.div = h
        self.n_dim = d_reduced
        
    def build(self, input_shape):
        self.multi = Multi_Head_Attention(self.div, input_shape[-1], self.n_dim)
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn_1 = tf.keras.layers.Dense(input_shape[-1]*4)
        self.ffn_2 = tf.keras.layers.Activation('relu')
        self.ffn_3 = tf.keras.layers.Dense(input_shape[-1])
        super(Encoder, self).build(input_shape)
        
    def call(self, x, training = False, mask = False):
        temp = self.multi([x, x, x])
        x = self.ln(x+temp)
        temp = self.ffn_1(x)
        temp = self.ffn_2(temp)
        temp = self.ffn_3(temp)
        x = self.ln(x+temp)
        return x


# In[102]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self, h, d_reduced):
        super(Decoder, self).__init__()
        self.div = h
        self.n_dim = d_reduced
        
    def build(self, input_shape):
        self.multi_1 = Multi_Head_Attention(self.div, input_shape[0][-1], self.n_dim)
        self.multi_2 = Multi_Head_Attention(self.div, input_shape[0][-1], self.n_dim)
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn_1 = tf.keras.layers.Dense(input_shape[0][-1]*4)
        self.ffn_2 = tf.keras.layers.Activation('relu')
        self.ffn_3 = tf.keras.layers.Dense(input_shape[0][-1])
        super(Decoder, self).build(input_shape)
        
    def call(self, inputs, training = False, mask = False):
        x, context = inputs
        temp = self.multi_1([x, x, x])
        x = self.ln(x+temp)
        temp = self.multi_2([x, context, context], mask = True)
        x = self.ln(x+temp)
        temp = self.ffn_1(x)
        temp = self.ffn_2(temp)
        temp = self.ffn_3(temp)
        return self.ln(x+temp)


# In[103]:


class Transformer(tf.keras.Model):
    def __init__(self, src, dst, d_emb, d_reduced, enc_count, dec_count, h):
        super(Transformer, self).__init__()
        self.enc_emb = tf.keras.layers.Embedding(src, d_emb)
        self.dec_emb = tf.keras.layers.Embedding(dst, d_emb)
        
        self.encoders = [Encoder(h, d_reduced) for _ in range(enc_count)]
        self.decoders = [Decoder(h, d_reduced) for _ in range(dec_count)]
        
        self.dense = tf.keras.layers.Dense(dst)
        
    def call(self, x):
        src, dst = x
        src = self.enc_emb(src)
        dst = self.dec_emb(dst)
        for layer in self.encoders:
            src = layer(src)
        for layer in self.decoders:
            dst = layer((dst, src))
        print(tf.keras.layers.Activation('softmax')(self.dense(dst)))
        return tf.keras.layers.Activation('softmax')(self.dense(dst))

