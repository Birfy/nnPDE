'''
Partial Derivative Equation Solver Using Neural Network

By Birfy
Dec.19 2018

Depencency: tensorflow, numpy
'''

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
        g = tf.gradients(grad[:,i], x)[0]
        delta += g[:,i]
    return delta

def compute_dxdx(u, x, d):
    '''
    Cauculate Second Derivative
    u-function
    x-variable
    d-direction
    '''
    grad = tf.gradients(u, x)[0]
    grad = tf.gradients(grad[:,d], x)[0]
    dudxdx = grad[:,d]
    return dudxdx

def compute_dx(u, x, d):
    '''
    Calculate Derivative
    u-function
    x-variable
    d-direction
    '''
    grad = tf.gradients(u, x)[0]
    dudx = grad[:,d]
    return dudx

class NNPDE:
    '''
    This NN model contains two networks to solve a PDE
    A-border equation
    B-border condition
    f-inner equation
    
    For border:
        Train bsubnetwork to get a approximate function
    For inner:
        Total function u = bsubnetwork + B * subnetwork

    Training Process:
        Train the bsubnetwork and subnetwork in turn
    '''

    def __init__(self, batch_size, N, d):
        self.d = d
        self.batch_size = batch_size
        self.N = N

        self.x = tf.placeholder(tf.float64, (None, d))
        self.x_b = tf.placeholder(tf.float64, (None, d))

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.loss = self.loss_function()
        self.bloss = self.bloss_function()

        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss,var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()

    def A(self, x):
        raise NotImplementedError

    def B(self, x):
        raise NotImplementedError
    
    def f(self, x):
        raise NotImplementedError
    
    def subnetwork(self, x, reuse=False):
        '''
        Network for inner
        '''
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 64, activation=tf.nn.tanh, name="dense{}".format(i),reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name='output', reuse=reuse)
            x = tf.squeeze(x, axis=1)
        return x
    
    def bsubnetwork(self, x, reuse = False):
        '''
        Network for border
        '''
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 64, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="boutput", reuse=reuse)
            x = tf.squeeze(x, axis=1)
        return x 

    def loss_function(self):
        assert NotImplementedError

    def bloss_function(self):
        assert NotImplementedError
    
    def train(self, sess, i):
        assert NotImplementedError

    def calculate(self, sess, X):
        '''
        Calculate u value at X
        '''
        value = sess.run(self.u, feed_dict={self.x: X})
        return value

class Problem2(NNPDE):

    def A(self, x):
        return x[:,0]

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0])
        # return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0

    def loss_function(self):
        deltah = compute_dxdx(self.u, self.x, 0)
        delta = self.f(self.x)
        res = tf.reduce_mean((deltah - delta) ** 2)
        return res

    def bloss_function(self):
        return tf.reduce_mean((self.A(self.x_b)-self.u_b)**2)

    def train(self, sess, i):

        bX = np.random.rand(2*self.batch_size, self.d)
        for j in range(1):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 0.0
            # if j==0:
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 1.0
        
        result, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})
        
        X = np.random.rand(self.batch_size, self.d)
        result, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})
        
        # bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary

        if i % 10 == 0:
        	# pass
            print("Iteration={}, bloss = {}, loss= {}".format(i, bloss, loss))

def draw():
    result = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            result[i][j] = npde.calculate(sess, [[i/100, j/100]])
    pyplot.imshow(result)
    pyplot.show()

npde = Problem2(512, 6, 2)
with tf.Session() as sess:
    sess.run(npde.init)
    for i in range(1000):
        npde.train(sess, i)
    draw()

    