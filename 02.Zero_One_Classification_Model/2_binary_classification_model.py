# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 00:15:09 2018

@author: LeeJY
"""

import tensorflow as tf
import pandas as pd

class binary_classifier :
    def __init__(self,sess):
        self.sess = sess 
        self.Num_of_Features = 9 #Input Layer 노드 개수 
        self.Hidden1_SIZE = 6 #1번째 Hidden node 개수
        self.Num_of_Output = 1 #Output Layer node 개수
        self.Hidden1_Act = None #Hidden Layer Node의 activation function 결과를 저장하는 변수
        self.hypothesis = None #Output Layer Node의 activation function 결과를 저장하는 변수
        self.cost = None #cost 값을 저장하는 변수
        self.optimization = None #Train에 대한 Optimization을 저장하는 변수
        self.prediction = None #예측 결과를 저장하는 변수
        self.accuracy = None #정확도를 저장하는 변수
        self.Max_Step = 1001

        self._build_net() #모델 생성하기 함수 호출
    
    def _build_net(self): #모델 생성하기
        with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=([None,self.Num_of_Features]),name="X")
            self.Y = tf.placeholder(tf.int32, shape=([None,self.Num_of_Output]),name="Y")

            self.Y_one_hot = tf.one_hot(self.Y, 2, name="Y_one_hot")
            self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1,2])
        
            self.W1 = tf.get_variable(name='W1',initializer=tf.truncated_normal([self.Num_of_Features,self.Hidden1_SIZE]))
        
            self.b1 = tf.get_variable(name='b1',initializer=tf.zeros([self.Hidden1_SIZE]))

            self.Hidden1_Act = tf.matmul(self.X, self.W1) + self.b1

            self.W2 = tf.get_variable(name='W2',initializer=tf.truncated_normal([self.Hidden1_SIZE,2]))
            
            self.b2 = tf.get_variable(name='b2',initializer=tf.zeros([2]))

            self.logits = tf.matmul(self.Hidden1_Act,self.W2)+self.b2

            self.hypothesis = tf.nn.softmax(self.logits)
            
            self.cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y_one_hot)

            self.cost = tf.reduce_mean(self.cost_i)
        
            self.optimization = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.cost)
                 
            self.prediction = tf.argmax(self.hypothesis, 1)

            self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y_one_hot,1))
            
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
            
    def _train_model(self,pixel_input,label_input): #모델 학습하기
        return self.sess.run([self.cost, self.accuracy], feed_dict={self.X: pixel_input, self.Y: label_input})
    
    def _prediction_(self,pixel_input,label_input): #학습된 모델을 이용하여 예측하기
        test_predict = self.sess.run([self.accuracy,self.prediction,self.correct_prediction], feed_dict={self.X: pixel_input, self.Y: label_input})

        return test_predict