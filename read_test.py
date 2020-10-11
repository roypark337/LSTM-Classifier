import pandas as pd
import re
import konlpy
from konlpy.tag import Okt
import numpy as np
import matplotlib.pyplot as plt

train_data= pd.read_csv('merge_training.csv',error_bad_lines=False)
test_data= pd.read_csv('merge_testing.csv',error_bad_lines=False)

train_data['script'] = train_data['script'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['script'] = test_data['script'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords=['의','제','위','그','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','것','위','수','등','관','함','점','및','대한']
okt = Okt()

print(len(train_data))
print(len(test_data))

#train_data['index'].value_counts().plot(kind='bar')
#test_data['index'].value_counts().plot(kind='bar')

print(train_data.groupby('index').size().reset_index(name='count'))
print(test_data.groupby('index').size().reset_index(name='count'))

X_train=[]
for sentence in train_data['script']:
    temp_X = []
    temp_X=okt.morphs(str(sentence), stem=True) # 토큰화
    temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)
           
X_test=[]
for sentence in test_data['script']:
    temp_X = []
    temp_X=okt.morphs(str(sentence), stem=True) # 토큰화
    temp_X=[word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)


from keras.preprocessing.text import Tokenizer
max_words = 35000
tokenizer = Tokenizer(num_words=max_words) # 상위 35,000개의 단어만 보존
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))

y_train=train_data['index']
y_test=test_data['index']

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

max_len=1500
# 전체 데이터의 길이는 1500으로 맞춘다.
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=1000, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
