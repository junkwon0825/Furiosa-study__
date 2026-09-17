from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D

model = Sequential()
model.add(Conv2D(50, (4,4), input_shape=(30,30,1))) # 필터 크기 (height, width, channel)
model.add(Conv2D(30, (3,3))) # 필터 크기

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 4, 4, 10)            │              50 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 3, 3, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 255 (1020.00 B)
#  Trainable params: 255 (1020.00 B)
#  Non-trainable params: 0 (0.00 B)




#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 18, 18, 20)        200       
                                                                 
#  conv2d_1 (Conv2D)           (None, 16, 16, 30)        5430      
                                                                 
# =================================================================
# Total params: 5,630
# Trainable params: 5,630

model.add(Conv2D(50, (4,4), input_shape=(30,30,1))) # 필터 크기 (height, width, channel)
model.add(Conv2D(30, (3,3))) # 필터 크기
_________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 27, 27, 50)        850       
                                                                 
#  conv2d_1 (Conv2D)           (None, 25, 25, 30)        13530     
                                                                 
# =================================================================
# Total params: 14,380
# Trainable params: 14,380
# Non-trainable params: 0