import tensorflow as tf

import pandas as pd

from binary_classification_model import binary_classifier


#테스트를 위한 데이터 값
test_pixel = [[1, 1, 1, 1, 0, 1, 1, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 0]]
 
test_label = [[0],[1]]

sess = tf.Session()

model = binary_classifier(sess) 

saver = tf.train.Saver()

saver.restore(sess,tf.train.latest_checkpoint('./models/')) #1501번 학습한 결과를 저장한 모델을 불러옴

prediction_result = model._prediction_(test_pixel,test_label) #불러온 모델을 이용하여 테스트 데이터를 사용하여 예측하기

print(prediction_result) #예측한 결과 출력하기
