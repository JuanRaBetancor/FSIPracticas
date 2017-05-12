import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]


# TODO: the neural net!!

x = tf.placeholder("float", [None,784])
y_ = tf.placeholder("float", [None,10])

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size=20
valid_error= 0.1;
train_errors = []
valid_errors = []
current_valid_error = 1
epoch = 0
current_diff = 100
test_errors = []

while(0.1 <= current_valid_error and current_diff > 0.001):
    epoch+=1
    for jj in xrange(len(train_x)/batch_size):
        batch_train_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_train_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_train_xs, y_: batch_train_ys})



    train_error = sess.run(loss, feed_dict={x: batch_train_xs, y_: batch_train_ys})
    train_errors.append(train_error)

    valid_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    valid_errors.append(valid_error)

    if(epoch > 1):
        current_diff = valid_errors[-2] - valid_error

    current_valid_error = valid_error

    # print "Entrenamiento---->"
    # print "Epoch #:", epoch, "Error: ", train_error
    # result = sess.run(y, feed_dict={x: batch_train_xs})
    # for b, r in zip(batch_train_ys, result):
    #     print b, "-->", r
    #
    # print "Validacion------->"
    # print "Epoch #:", epoch, "Error: ", current_valid_error
    # result = sess.run(y, feed_dict={x: batch_valid_xs})
    # for b, r in zip(batch_valid_ys, result):
    #     print b, "-->", r

numAciertos = 0
numFallos = 0

print "Test------->"
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if (np.argmax(b) != np.argmax(r)):
        numFallos += 1
    else:
        numAciertos += 1

print "Numero de aciertos en el Test: ", numAciertos
print "Numero de fallos en el Test: ", numFallos

plt.ylabel("Error")
plt.xlabel("Epochs")

graph_train, = plt.plot(train_errors)

plt.legend(handles=[graph_train],
labels=["Training errors"])
plt.savefig('mnist.png')