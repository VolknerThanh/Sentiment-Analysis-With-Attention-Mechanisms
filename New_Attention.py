import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os, re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from string import punctuation
from numpy import asarray
from os import listdir
from numpy import zeros
from numpy import array
from keras.layers import Input, Embedding, GRU, Dense, Dot
from keras.models import Model
from keras.layers.core import *
from keras import backend as K
from keras.layers import merge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('9'):
            continue
        if not is_trian and not filename.startswith('9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix





# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
print(len(vocab))

np.random.seed(1000)
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 200
epochs = 10

# load all training reviews
positive_docs = process_docs('dataset_Attention/book/pos', vocab, True)
negative_docs = process_docs('dataset_Attention/book/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs_train = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
print(max_length)
Xtrain = pad_sequences(encoded_docs_train, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

print(len(Xtrain))
# load all test reviews
positive_docs = process_docs('dataset_Attention/book/pos', vocab, False)
negative_docs = process_docs('dataset_Attention/book/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs_test = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs_test, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])


# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
num_most_freq_words_to_include = vocab_size
print(vocab_size)

# load embedding from file
raw_embedding = load_embedding('glove.6B/glove.6B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
print('weight' + str(embedding_vectors.shape))
# create the embedding layer
embedding_layer = Embedding(num_most_freq_words_to_include, 100, weights=[embedding_vectors], input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, trainable=False)



def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    attention = Dense(1, activation='tanh')(inputs)                             # input shape = batch * time_steps * 1
    attention = Flatten()(attention)                                            # input shape = batch * time_steps
    attention = Activation('softmax')(attention)                                # input shape = batch * time_steps
    attention = RepeatVector(input_dim)(attention)                              # input shape = batch * input_dim * time_steps
    attention = Permute([2, 1])(attention)                                      # input shape = batch * time_step * input_dim
    # sent_representation = merge([inputs, attention], mode='mul')                # input shape = batch * time_step * input_dim
    sent_representation = merge.multiply([inputs, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),               # input shape = batch * input_dim
                                 output_shape=(input_dim,))(sent_representation)
    return sent_representation


## the rnn model for sentiment analysis
def rnn_model():
    input_sequences = Input(shape=(MAX_REVIEW_LENGTH_FOR_KERAS_RNN,))
    embedding_layers = embedding_layer
    embout = embedding_layers(input_sequences)
    gruout = GRU(100, return_sequences=True)(embout)
    attout = attention_3d_block(gruout)
    outputs = Dense(1, activation='sigmoid')(attout)
    model = Model(inputs=input_sequences, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


## RNN model initization
gru_att_model = rnn_model()
gru_att_model.summary()


## training the rnn model
h = gru_att_model.fit(Xtrain, ytrain, batch_size=64, epochs=epochs, validation_data=(Xtest, ytest), verbose=2)
y_test_pred_gru_att = gru_att_model.predict(Xtest)
#
# ## calculatin the accuracy and f1 score
# print("The AUC socre for GRU attention model is : %.4f." %roc_auc_score(ytest, y_test_pred_gru_att.round()))
# print("F1 score for GRU attention model is: %.4f." % f1_score(ytest, y_test_pred_gru_att.round()))

loss_acc = gru_att_model.evaluate(Xtest, ytest, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
      (loss_acc[0], loss_acc[1]*100))

test_1 = "this book is fantastic! I really like it because it is so good!"
test_2 = "good book"
test_3 = "Maybe I like this book!"
test_4 = "if you like action, then this book might be good for you"
test_5 = "bad book"
test_6 = "this book is really suck! Can I get my money back please?"

test_samples = "this does not look like a nice book"
# test_samples = clean_doc(test_samples, vocab)
# tokenizer.fit_on_texts(test_samples)
test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN, padding='post')
print(test_samples_tokens)

prediction = gru_att_model.predict(x=test_samples_tokens_pad)
print("Prediction (0 = negative, 1 = positive) = ")
print("%0.4f" % prediction[0][0])


# print("Saving model to disk \n")
# mp = "Models/book_model.h5"
# gru_att_model.save(mp)

# serialize model to JSON
# model_json = gru_att_model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# gru_att_model.save_weights("Models/book_model.h5")
# print("Saved model to disk")

import matplotlib.pyplot as plt
from keras.layers import *
N = epochs
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), h.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), h.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), h.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), h.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()

# plt.subplot(211)
# plt.title('Loss')
# plt.plot(h.history['loss'], label='train')
# plt.plot(h.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# # # plot accuracy during training
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(h.history['acc'], label='train')
# plt.plot(h.history['val_acc'], label='test')
# plt.legend()
# plt.show()

