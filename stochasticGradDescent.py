from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import randint


import argparse
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with np.load("tinymnist.npz") as data :
	trainData, trainTarget = data["x"], data["y"]
	validData, validTarget = data ["x_valid"], data["y_valid"]
	testData, testTarget = data["x_test"], data["y_test"]


# useful dimension values
sample_size, dimension = trainData.shape

# Create the model
x = tf.placeholder(tf.float32, [None, dimension])
W = tf.Variable(tf.zeros([dimension, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b

def train(lamb, batch_size, learning_rate, predictor, label):
	losses = []

	# for loss function
	target = tf.placeholder(tf.float32, [None, 1]) #
	
	l2_loss = tf.nn.l2_loss(y-target) 
	regularizer = tf.nn.l2_loss(W)

	l2_loss_reg = l2_loss + lamb * regularizer

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss_reg)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# train
	for epoch in range (300):
		batch_start = randint(0, sample_size - batch_size + 1)
		batch_end = batch_start + batch_size

		batch_xs = predictor[batch_start : batch_end]
		batch_ys = label[batch_start : batch_end]

		_, loss, weights, bias, pred, reg = sess.run([train_step, l2_loss_reg, W, b, y, regularizer], 
			feed_dict={x: batch_xs, target: batch_ys})
		losses.append(loss)

		if (epoch % 50 == 0):
			print("epoch: %d\t; reg: %f\t; L2 loss = %f" % (epoch, reg, loss))
		# print(tf.transpose(W))
		

	# Test trained model
	correct_prediction = tf.equal(tf.round(y), target)
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# print(sess.run(accuracy, feed_dict={x: testData, target: testTarget}))
	accuracy = sess.run(acc, feed_dict={x: predictor, target: label})

	return losses, weights, bias, accuracy
	# return loss
	# return accuracy


def plotBatchVsLoss():
	lamb = 1
	losses = []
	batch_sizes = [10, 50, 100, 700]
	learning_rates = [.03, .006, .003, .0002]

	for i in range(4):
		loss, _, _ = train(lamb, batch_sizes[i], learning_rates[i])
		losses.append(loss)
		# print("size = %d rate = %f loss = %f" % (batch_size, learning_rate, loss))

	plt.figure(1)

	plt.subplot(221)
	plt.plot(losses[0])
	plt.title('Batch size: 10; learning rate: 0.03')

	plt.subplot(222)
	plt.plot(losses[1])
	plt.title('Batch size: 50; learning rate: 0.006')

	plt.subplot(223)
	plt.plot(losses[2])
	plt.title('Batch size: 100; learning rate: 0.003')

	plt.subplot(224)
	plt.plot(losses[3])
	plt.title('Batch size: 700; learning rate: 0.0002')

	# plt.plot(losses)
	# plt.ylabel('L2 Loss')
	# plt.xlabel('epochs')
	# plt.xlim((1,700))
	# plt.title('Loss vs Epoch. Learning rate = 0.006')
	plt.show()


def test(weights, bias, predictor, label):
	prediction = tf.matmul(x, weights) + bias

	sess = tf.Session()
	pred = sess.run(prediction, feed_dict={x: predictor})

	correct_prediction = tf.equal(tf.round(pred), label)
	pred_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	accuracy = sess.run(pred_accuracy)
	return accuracy


# plotBatchVsLoss()
lambs = [0., .0001, .001, .01, .1, 1.]
batch_size = 700
learning_rate = .0002

accuracies = []

for lamb in lambs:
	# train a model using different lambda values
	_, weights, bias, train_accuracy = train(lamb, batch_size, learning_rate, trainData, trainTarget)

	# test result on the cross validation set
	validation_accuracy = test(weights, bias, validData, validTarget)

	accuracies.append(validation_accuracy)
	print("lambda = %f validation_accuracy = %f" % (lamb, validation_accuracy))

plt.figure(2)
plt.plot(lambs, accuracies)
plt.title('Lambda vs Accuracy')
plt.xscale('log')
plt.show()
