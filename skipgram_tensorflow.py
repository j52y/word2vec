from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import *

'''train with softmax and cross entropy loss
'''

sentences = read_corpus()
words = words(sentences)
w2i, i2w = word_index(words)
voca_size = len(i2w)

train_x, train_y = train_data(words, w2i, i2w)
t_size = len(train_x)

train_x = one_hot_multiple(voca_size, train_x)
train_y = one_hot_multiple(voca_size, train_y)

x = tf.placeholder(tf.float32, (t_size, voca_size))
y = tf.placeholder(tf.float32, (t_size, voca_size))

embedd_size = 30
learnig_rate = 1

sess = tf.InteractiveSession()

W1 = tf.Variable(tf.random_uniform((voca_size, embedd_size), -1, 1))
W2 = tf.Variable(tf.random_uniform((embedd_size, voca_size), -1, 1))

sess.run(tf.initialize_all_variables())

h = tf.matmul(x, W1, a_is_sparse=True)
O = tf.matmul(h, W2)
softmax = tf.nn.softmax(O)
mul = tf.mul(tf.log(softmax), y)
loss = tf.reduce_mean(-mul)
train = tf.train.GradientDescentOptimizer(learnig_rate).minimize(loss)

iteration = int(1e6)

for i in range(iteration):
    train.run(feed_dict={x: train_x, y: train_y})

    if i % 1e3 == 0:
        error = loss.eval(feed_dict={x: train_x, y: train_y})
        print('iteration: %d, loss: %f' % (i, error))

        w1 = W1.eval()
        word = 'artificial'
        sims = cos_sim(w1)
        row = sims[w2i[word]].argsort()[::-1][:5]
        k = [(i2w[i], sims[w2i[word], i]) for i in row]
        print('similar words with %s: %s' % (word, k))

print('finished. loss: %f' % loss.eval(feed_dict={x: train_x, y: train_y}))
