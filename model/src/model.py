'''
Created on 14 Nov 2018

@author: tuanluu
'''

import tensorflow as tf
import numpy as np

class Model(object):
    

    def __init__(self, num_classes, vocab_size, embedding_size, learning_rate, init_value):
        # Placeholders for input, output and dropout
        self.input_x_pre = tf.placeholder(tf.int32, [None, None], name="input_x_pre")
        self.input_x_post = tf.placeholder(tf.int32, [None, None], name="input_x_post")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.mask_pre = tf.sign(tf.abs(self.input_x_pre))
        self.mask_pre = tf.expand_dims(self.mask_pre,-1)
        self.mask_pre = tf.cast(self.mask_pre,tf.float32)
        
        self.mask_post = tf.sign(tf.abs(self.input_x_post))
        self.mask_post = tf.expand_dims(self.mask_post,-1)
        self.mask_post = tf.cast(self.mask_post,tf.float32)
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -init_value, init_value),trainable=True, name="W", dtype = tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size], name="embedding_placeholder")
            self.embedding_init = W.assign(self.embedding_placeholder)
        
        with tf.name_scope("concat_vectors"):
            self.train_input_embedding_left = tf.nn.embedding_lookup(W, self.input_x_pre)
            self.train_input_embedding_left = self.train_input_embedding_left * self.mask_pre
#             temp = tf.fill([tf.shape(self.train_input_embedding_left)[0],1],1.0)
#             self.train_input_embedding_left = tf.divide(tf.reduce_sum(self.train_input_embedding_left,1),self.sentence_length(self.train_input_embedding_left))
            self.train_input_embedding_left = tf.reduce_mean(self.train_input_embedding_left,1)
            
            self.train_input_embedding_right = tf.nn.embedding_lookup(W, self.input_x_post)
            self.train_input_embedding_right = self.train_input_embedding_right * self.mask_post
#             temp = tf.fill([tf.shape(self.train_input_embedding_right)[0],1],1.0)
#             self.train_input_embedding_right = tf.divide(tf.reduce_sum(self.train_input_embedding_right,1),self.sentence_length(self.train_input_embedding_right))
            self.train_input_embedding_right = tf.reduce_mean(self.train_input_embedding_right,1)
            
            self.train_input_embedding = tf.concat([self.train_input_embedding_left, self.train_input_embedding_right,
                                                    tf.add(self.train_input_embedding_left, self.train_input_embedding_right),
                                                    tf.subtract(self.train_input_embedding_left, self.train_input_embedding_right),
                                                    tf.multiply(self.train_input_embedding_left, self.train_input_embedding_right)], 1)
        
        
        with tf.name_scope("output"):
            W1 = tf.get_variable(
                "W1",
                shape=[5*embedding_size, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.01, shape=[embedding_size]), name="b")
            self.scores1 = tf.nn.xw_plus_b(self.train_input_embedding, W1, b1, name="scores1")
            self.layers1 = tf.nn.tanh(self.scores1)
            self.dropout_1 = tf.nn.dropout(self.layers1,self.dropout_keep_prob)
            
            W2 = tf.get_variable(
                "W2",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.01, shape=[num_classes]), name="b")
            self.scores = tf.nn.tanh(tf.nn.xw_plus_b(self.dropout_1, W2, b2, name="scores"))
            
            self.pred_ops = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) #+ l2_reg_lambda * l2_loss
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads_and_vars)
  
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    
    def sentence_length(self,sequence):
        '''
        Find the actual length of a sequence
        Input: sequence has size of [batch_size,sequence_length,embedding_size]
        Output: actual length of the sequence (exclude the padding elements)
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.float32)
        return length    
    
