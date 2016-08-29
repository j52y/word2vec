from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

from keras.models import Sequential
from keras.layers import Dense, Activation

sentences = read_corpus()
words = words(sentences)
w2i, i2w = word_index(words)
v = len(i2w)

train_x, train_y = train_data(words, w2i, i2w)
t_size = len(train_x)

train_x = one_hot_multiple(v, train_x)
train_y = one_hot_multiple(v, train_y)

n = 30  # embedd size

model = Sequential()

model.add(Dense(output_dim=n, input_dim=v))
model.add(Dense(output_dim=v))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

iteration = 100000
for i in range(iteration):
    model.train_on_batch(train_x, train_y)
    if i % 1000 == 0:
        W0 = model.get_weights()[0]
        out = cos_sim(W0)
        word = 'artificial'
        sims = out[w2i[word]].argsort()
        sims = sims[::-1][:5]
        loss_and_metrics = model.evaluate(train_x, train_y, batch_size=32)
        print("\nsims: %s" % [(i2w[i], out[w2i[word], i]) for i in sims])
        print("loss: %f" % loss_and_metrics[0])

loss_and_metrics = model.evaluate(train_x, train_y, batch_size=32)
print("loss: %f" % loss_and_metrics[0])
