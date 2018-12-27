import tensorflow as tf
import numpy as np
from matplotlib import pyplot

def compute_delta(u, x, n):
    '''
    Calculate Laplace Oprator
    u-function
    x-variable
    n-number of variables
    '''
    grad = tf.gradients(u, x)[0]
    delta = 0
    for i in range(n):
        g = tf.gradients(grad[:, i], x)[0]
        delta += g[:, i]
    return delta


def compute_dxdx(u, x, d):
    '''
    Cauculate Second Derivative
    u-function
    x-variable
    d-direction
    '''
    grad = tf.gradients(u, x)[0]
    grad = tf.gradients(grad[:, d], x)[0]
    dudxdx = grad[:, d]
    return dudxdx


def compute_dx(u, x, d):
    '''
    Calculate Derivative
    u-function
    x-variable
    d-direction
    '''
    grad = tf.gradients(u, x)[0]
    dudx = grad[:, d]
    return dudx

class UPDES:

    def __init__(self, batch_size, units, layers, dimensions):
        self.batch_size = batch_size
        self.units = units
        self.layers = layers
        self.dimensions = dimensions

        self.x = tf.placeholder(tf.float64, (None, dimensions))
        self.u = self.subnetwork(self.x)

        self.loss = self.loss_function()
        self.bloss = self.bloss_function()

        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.bopt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss)

        self.init = tf.global_variables_initializer()

    def B(self, x):
        pass
    
    def I(self, x):
        pass

    def subnetwork(self, x):
        for i in range(self.layers):
            x = tf.layers.dense(x, self.units, activation=tf.nn.tanh, name="dense{}".format(i))
        x = tf.layers.dense(x, 1, activation=None, name='output')
        x = tf.squeeze(x, axis=1)
        return x

    def loss_function(self):
        pass

    def bloss_function(self):
        pass

    def train(self):
        pass

    def calculate(self, sess, X):
        value = sess.run(self.u, feed_dict={self.x: X})
        return value

class HeatTransfer(UPDES):
    def B(self, x):
        return x[:, 0] * tf.sin(np.pi * x[:, 1])

    def I(self, x):
        return 0

    def loss_function(self):
        laplace = compute_delta(self.u, self.x, 2)
        res = tf.reduce_mean((laplace - self.I(self.x))**2)
        return res

    def bloss_function(self):
        return tf.reduce_mean((self.B(self.x) - self.u)**2)
    
    def train(self, sess, i):
        # bX = np.random.rand(4 * self.batch_size, self.dimensions)
        # for j in range(2):
        #     bX[2 * j * self.batch_size:(2 * j + 1) * self.batch_size, j] = 0.0
        #     # if j==0:
        #     bX[(2 * j + 1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 1.0
        # bloss = sess.run([self.bloss], feed_dict={self.x: bX})
        
        # X = np.random.rand(self.batch_size, self.dimensions)
        # loss = sess.run([self.loss], feed_dict={self.x: X})[0]

        
        
        bX = np.random.rand(4 * self.batch_size, self.dimensions)
        for j in range(2):
            bX[2 * j * self.batch_size:(2 * j + 1) * self.batch_size, j] = 0.0
            # if j==0:
            bX[(2 * j + 1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 1.0
        result, bloss = sess.run([self.bopt, self.bloss], feed_dict={self.x: bX})
        # print(bloss)
    
    # while loss > bloss:
        X = np.random.rand(self.batch_size * 4, self.dimensions)
        result, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
        # print(loss)
            # print(loss)
        
        # if the loss is small enough, stop training on the boundary

        # if i % 10 == 0:
            # pass
        print("Iteration = {}, bloss = {:.10f}, loss = {:.10f}".format(i, bloss, loss))

def draw():
    result = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            result[i][j] = npde.calculate(sess, [[i / 100, j / 100]])
            # print("x : {}, y : {}, u : {}".format(i, j, result[i][j]))
    # pyplot.contour(result,20)
    pyplot.imshow(result)
    pyplot.show()


npde = HeatTransfer(512, 5, 3, 2)
with tf.Session() as sess:
    sess.run(npde.init)
    for i in range(10000):
        npde.train(sess, i)
    draw()