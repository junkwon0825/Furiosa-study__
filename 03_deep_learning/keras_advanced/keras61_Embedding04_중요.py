#61-3 copy
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM ,Embedding, SimpleRNN
import time
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#1.data
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미 없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

print(x)


### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_x = pad_sequences(
    x,
    maxlen=5,
    padding='pre',
    truncating='post'
)

print(padded_x)
print(padded_x.shape)      # (15, 5)


y = labels[0]


#2.model
model = Sequential()
############################### EMBEDDING 1 ########################
model.add(Embedding(input_dim=31, output_dim=100, input_length=5)) #단어사전의 갯수, 차원
model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100      
                                                                 
#  simple_rnn (SimpleRNN)      (None, 10)                1110      
                                                                 
# =================================================================


############################### EMBEDDING 2 ########################
model.add(Embedding(input_dim=31, output_dim=100)) #input_length 명시안해도 알아서 맞춰줌
model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()

############################### EMBEDDING 3 ########################
model.add(Embedding(31, 100)) #input_dim,output_dim 명시안해도 알아서 맞춰줌

model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()

#3.compile,train
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=500,
    restore_best_weights=True
)

start_time = time.time()

history = model.fit(
    x,
    y,
    epochs=1000,
    batch_size=5,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

train_time = time.time() - start_time


#