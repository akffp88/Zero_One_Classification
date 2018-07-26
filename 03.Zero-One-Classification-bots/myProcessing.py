# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:50:08 2018

@author: LEE
"""

import tensorflow as tf
import sys
import os
from binary_classification_model import binary_classifier

test_0_pixel = [[1, 1, 1, 1, 0, 1, 1, 1, 1]]
test_0_label = [[0]]
test_1_pixel = [[0, 1, 0, 0, 1, 0, 0, 1, 0]]
test_1_label = [[1]]

model = dict()

def _setup_():
    global model, test_0_label, test_0_pixel, test_1_label, test_1_pixel
    
    sess = tf.Session()
    
    model = binary_classifier(sess)
    
    saver = tf.train.Saver()
    
    saver.restore(sess,tf.train.latest_checkpoint('./models'))
    
def _get_response_(content):   
    if content == 0 :
        result = model._prediction_(test_0_pixel,test_0_label)
    else :
        result = model._prediction_(test_1_pixel,test_1_label)
    return result
