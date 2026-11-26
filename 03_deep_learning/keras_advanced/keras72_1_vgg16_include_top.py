import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16

#2. model
# model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
# #include_top false로 하면 input shape 알아서 바꿀수있음
# model.summary()

#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         

#  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 100, 100, 3)]     0         
                                                                 
#  block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688

############# include_top = False ###############
#1. FC layer 없어짐 ( 직접 아래에 fc layer를 명시 해주면 됨)
#2. input_shape 를 원하는 데이터 shape로 변경




