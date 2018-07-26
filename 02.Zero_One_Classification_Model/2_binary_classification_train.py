# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 00:30:25 2018

@author: LeeJY
"""

import tensorflow as tf

import numpy as np

from binary_classification_model import binary_classifier
#binary_classification_model.py에서 생성한 binary_classifier 클래스를 추가

input_data = np.loadtxt("data.txt",dtype=float,delimiter=',')

pixel_data = input_data[:,0:-1]

label_data = input_data[:,[-1]]

sess = tf.Session() 

model = binary_classifier(sess) #binary_classifier 클래스에 대한 객체 생성

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver() #Tensorflow 모델을 저장하기 위한 변수

for step in range(1501):
    c_, a_ = model._train_model(pixel_data,label_data)
    if step % 100 == 0:
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, c_, a_))
    
saver.save(sess,"./models/",global_step=1501) #1501번 학습한 모델을 저장
