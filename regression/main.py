#!/usr/bin/env 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


COLUMN_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
				'TAX', 'PTRATIO', 'B', 'LSTAT']

EPOCHS = 500

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):

	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')

class Regression(object):
	def __init__(self):
		self.train_data = []
		self.train_labels = []
		self.test_data = []
		self.test_labels = []
		self.model = []

	def load_dataset(self):
		boston_housing = keras.datasets.boston_housing
		(self.train_data, self.train_labels), (self.test_data, self.test_labels) = boston_housing.load_data()
		print("Training set: {}".format(self.train_data.shape))  # 404 examples, 13 features
		print("Testing set:  {}".format(self.test_data.shape), end='\n\n')   # 102 examples, 13 features

	def display_dataset(self):
		df = pd.DataFrame(self.train_data, columns=COLUMN_NAMES)
		print(df.head(), end='\n\n')

	def preprocess(self):
		mean = self.train_data.mean(axis=0)
		std = self.train_data.std(axis=0)
		self.train_data = (self.train_data - mean) / std
		self.test_data = (self.test_data - mean) / std

		print('Normalized Tranding data: ', self.train_data[0], end='\n\n')

	def build_model(self):
		model = keras.Sequential([
			keras.layers.Dense(64, activation=tf.nn.relu,
							input_shape=(self.train_data.shape[1],)),
			keras.layers.Dense(64, activation=tf.nn.relu),
			keras.layers.Dense(1)
		])

		optimizer = tf.train.RMSPropOptimizer(0.001)

		model.compile(loss='mse',
						optimizer=optimizer,
						metrics=['mae'])
		self.model = model
		self.model.summary()

	def train(self):
		# The patience parameter is the amount of epochs to check for improvement
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

		history = self.model.fit(self.train_data, self.train_labels, epochs=EPOCHS,
					validation_split=0.2, verbose=0,
					callbacks=[PrintDot()])

		self.plot_history(history)

		[loss, mae] = self.model.evaluate(self.test_data, self.test_labels, verbose=0)

		print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

	def plot_history(self, history):
		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Abs Error [1000$]')
		plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
				label='Train Loss')
		plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
				label = 'Val loss')
		plt.legend()
		plt.ylim([0, 5])
		plt.savefig('assets/history.png')

	def predict(self):
		test_predictions = self.model.predict(self.test_data).flatten()

		plt.figure()
		plt.scatter(self.test_labels, test_predictions)
		plt.xlabel('True Values [1000$]')
		plt.ylabel('Predictions [1000$]')
		plt.axis('equal')
		plt.xlim(plt.xlim())
		plt.ylim(plt.ylim())
		_ = plt.plot([-100, 100], [-100, 100])
		plt.savefig('assets/predictions.png')

		plt.figure()
		error = test_predictions - self.test_labels
		plt.hist(error, bins = 50)
		plt.xlabel("Prediction Error [1000$]")
		_ = plt.ylabel("Count")

		plt.savefig('assets/prediction-error.png')

if __name__ == "__main__":
	regression = Regression()
	regression.load_dataset()
	regression.display_dataset()
	regression.preprocess()
	regression.build_model()
	regression.train()
	regression.predict()
	
	# regression.preprocess()
	# regression.train()
	# regression.predict()