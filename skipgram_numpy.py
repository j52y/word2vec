from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

sentences = read_corpus()
words = words(sentences)
w2i, i2w = word_index(words)
v = len(i2w)

train_x, train_y = train_data(words, w2i, i2w)
t_size = len(train_x)

h = 0.5  # learning rate
n = 30  # embedd size

np.random.seed(1)

W0 = np.random.random((v, n))
W1 = np.random.random((n, v))

iteration = 2000

for it in range(iteration):
    loss = 0.
    for i in range(t_size):
        tx, ty = train_x[i], train_y[i]

        # Since input tx is represented as one-hot encoding, x.dot(W0) is a n-th row of W0
        hidden = W0[tx]
        u = hidden.dot(W1)
        y = softmax(u)
        loss += -np.log(y[ty])

        # backpropagation
        dEdy = cross_entropy_one_hot_derivative(y, ty)
        dEdz = softmax_derivative(y).dot(dEdy)
        dEdW1 = matrix_multi_derivative(W1, hidden, dydx=False, x_is_multiplier=False).dot(dEdz)
        dEdW1 = dEdW1.reshape(W1.shape[0], W1.shape[1])

        dEdh = matrix_multi_derivative(W1, hidden, dydx=True, x_is_multiplier=False).dot(dEdz)
        dEdW0 = matrix_multi_derivative(W0, one_hot(v, tx).reshape(v, ), dydx=False, x_is_multiplier=False).dot(dEdh)
        dEdW0 = dEdW0.reshape(W0.shape[0], W0.shape[1])
        W1 -= h * dEdW1
        W0 -= h * dEdW0

    out = cos_sim(W0)
    word = 'artificial'
    sims = out[w2i[word]].argsort()
    sims = sims[::-1][:5]
    print("sims: %s" % [(i2w[i], out[w2i[word], i]) for i in sims])
    print("loss: %f" % (loss / (t_size * v)))
