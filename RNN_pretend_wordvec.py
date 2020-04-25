import csv
import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense, GRU, Dropout, SimpleRNN, LSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle

#------------------------- Read data ------------------------------
file = open('Dataset.csv', 'r',encoding = 'utf-8-sig')
data = list(csv.reader(file))
shuffle(data)
print(data[0])
labels = [int(d[0]) for d in data]

sentences = [d[1] for d in data]
print("tokenizing...")
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
print("tokenize complete")
max_sentence_length = max([len(s) for s in words])
print(max_sentence_length)
#------------------- Extract word vectors -------------------------
print("extracting word2vec...")
vocab = set([w for s in words for w in s])

pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
count = 0
vocab_vec = {}
for line in pretrained_word_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab):
            vocab_vec[line[0]] = line[1:]
    count = count + 1

word_vector_length = 300
word_vectors = np.zeros((len(words),max_sentence_length,word_vector_length))
sample_count = 0
for s in words:
    word_count = 0
    for w in s:
        try:
            word_vectors[sample_count,max_sentence_length-word_count-1,:] = vocab_vec[w]
            word_count = word_count+1
        except:
            pass
    sample_count = sample_count+1
print("extract word2vec complete")
# --------------- Create recurrent neural network-----------------
inputLayer = Input(shape=(max_sentence_length,word_vector_length,))
#hiddenLayer
rnn = GRU(100, activation='relu')(inputLayer)
h1 = Dense(100, activation='relu')(rnn)
h2 = Dense(10 ,activation='relu')(h1)
h2 = Dropout(0.25)(h2)
outputLayer = Dense(3, activation='softmax')(h2) #for 3 classes
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

#----------------------- Train neural network-----------------------
history = model.fit(word_vectors, to_categorical(labels), epochs=200, batch_size=64,verbose=1,validation_split=0.085)
#-------------------------- Evaluation-----------------------------

model.save('Saved_model/my_model.h5')  #save model

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
