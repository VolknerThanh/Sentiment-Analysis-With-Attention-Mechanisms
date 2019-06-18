from keras.layers import Input, Embedding, GRU, Dense, Dot
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import sequence
from keras.layers.core import *
from keras import backend as K
from keras.layers import merge
from keras.datasets import imdb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

np.random.seed(1000)
max_features = 5000
# cut texts after this number of words (among top max_features most common words)
maxlen = 200
batch_size = 32
epochs = 5
embedding_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)




## attention layer for weightage the generated hidden vector for sentiment prediction
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
    input_sequences = Input(shape=(maxlen,))
    embedding_layer = Embedding(input_dim=max_features,
                                output_dim=embedding_size,
                                input_length=maxlen)
    embout = embedding_layer(input_sequences)
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
# h = gru_att_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
# y_test_pred_gru_att = gru_att_model.predict(x_test)

## calculatin the accuracy and f1 score
# print("The AUC socre for GRU attention model is : %.4f." %roc_auc_score(y_test, y_test_pred_gru_att.round()))
# print("F1 score for GRU attention model is: %.4f." % f1_score(y_test, y_test_pred_gru_att.round()))

# print("Saving model to disk \n")
# mp = "Models/imdb_model.h5"
# gru_att_model.save(mp)

model = load_model("Models/imdb_model.h5")
print("New review: \'this is the best movie ever\'")
d = imdb.get_word_index()
review = "this is the best movie"

words = review.split()
review = []
for word in words:
  if word not in d:
    review.append(2)
  else:
    review.append(d[word]+3)

review = sequence.pad_sequences([review], maxlen=maxlen)
prediction = model.predict(review)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])

import matplotlib.pyplot as plt
from keras.layers import *
# N = epochs
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