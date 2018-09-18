#!/usr/bin/env 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class BasicClassification(object):
	def __init__(self):
		self.train_images = []
		self.train_labels = []
		self.test_images = []
		self.test_labels = []
		self.model = []
		self.fig = plt.figure()

	def load_dataset(self):
		fashion_mnist = keras.datasets.fashion_mnist
		(self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

	def preprocess(self):
		print('===== Start preprocessing =====')
		self.train_images = self.train_images / 255.0
		self.test_images = self.test_images / 255.0

		# show the trianing images and the corresponding labels
		plt.figure(figsize=(10,10))
		for i in range(25):
			plt.subplot(5,5,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(self.train_images[i], cmap=plt.cm.binary)
			plt.xlabel(CLASS_NAMES[self.train_labels[i]])
		plt.savefig('assets/traning-images.png')
		print('===== End up preprocessing =====')
	
	def train(self):
		''''
		keras.layers.Flatten(input_shape=(28, 28)):
			transforms the format of the images from a 2d-array (of 28 by 28 pixels),
			to a 1d-array of 28 * 28 = 784 pixels. only reformats the data.

		keras.layers.Dense():
			densely-connected.
			1st layer has 128 nodes (or neurons).
			2nd layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
		'''
		
		self.model = keras.Sequential([
			keras.layers.Flatten(input_shape=(28, 28)),
			keras.layers.Dense(128, activation=tf.nn.relu),
			keras.layers.Dense(10, activation=tf.nn.softmax)
		])

		self.model.compile(optimizer=tf.train.AdamOptimizer(), 
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])
	
		print('===== Start training =====')
		self.model.fit(self.train_images, self.train_labels, epochs=5)
		print('===== End up training =====')

		test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
		print('===== Evaluate =====')
		print('Test accuracy:', test_acc)
		print('Test loss:', test_loss)

	# def predict(self, test_images = self.test_images):
	# 	predictions = self.model.predict(test_images)

	def plot_image(self, i, predictions_array, true_label, img):
		predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])
		
		plt.imshow(img, cmap=plt.cm.binary)

		predicted_label = np.argmax(predictions_array)
		if predicted_label == true_label:
			color = 'blue'
		else:
			color = 'red'
		
		plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
										100*np.max(predictions_array),
										CLASS_NAMES[true_label]),
										color=color)
		plt.savefig('assets/test-images-{}-result.png'.format(i))

	def plot_value_array(self, i, predictions_array, true_label):
		predictions_array, true_label = predictions_array[i], true_label[i]
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])
		thisplot = plt.bar(range(10), predictions_array, color="#777777")
		plt.ylim([0, 1]) 
		predicted_label = np.argmax(predictions_array)
		
		thisplot[predicted_label].set_color('red')
		thisplot[true_label].set_color('blue')
		plt.savefig('assets/value-array.png')

	def predict(self, image_num = 0):
		predictions = self.model.predict(self.test_images)
		plt.figure(figsize=(6,3))
		plt.subplot(1,2,1)
		self.plot_image(image_num, predictions, self.test_labels, self.test_images)
		plt.subplot(1,2,2)
		self.plot_value_array(image_num, predictions, self.test_labels)
	

if __name__ == "__main__":
	classifier = BasicClassification()
	classifier.load_dataset()
	classifier.preprocess()
	classifier.train()
	classifier.predict()

