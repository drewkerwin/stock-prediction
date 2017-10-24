from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import math

# to not display the warnings of tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

	
# convert an array of values into a time series dataset 
# in form 
#                     X                     Y
# t-look_back+1, t-look_back+2, ..., t     t+1

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
		
def run_test():
	# parameters to be set ("optimum" hyperparameters obtained from grid search):
	look_back = 20
	epochs = 250
	batch_size = 512

	# fix random seed for reproducibility
	np.random.seed(7)

	# read all prices using panda
	prices_dataset =  pd.read_csv('data/spy.csv', header=0)

	# save Apple's stock values as type of floating point number
	spy_prices = prices_dataset.Adj_Close.values.astype('float32')

	# reshape to column vector
	spy_prices = spy_prices.reshape(len(spy_prices), 1)
	
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	spy_prices = scaler.fit_transform(spy_prices)
	
	# split data into training set and test set
	train_size = int(len(spy_prices) * 0.60)
	test_size = len(spy_prices) - train_size
	train, test = spy_prices[0:train_size,:], spy_prices[train_size:len(spy_prices),:]

	print('Split data into training set and test set... Number of training samples/ test samples:', len(train), len(test))

	# convert Apple's stock price data into time series dataset
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	
	# reshape input of the LSTM to be format [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(32, input_shape=(look_back, 1), activation='linear'))
	model.add(Dense(1))

	# compile model
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	# train the model; save model file when improved
	checkpointer = ModelCheckpoint(filepath='saved_models/money.hdf5', verbose=1, save_best_only=True)
	history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
		validation_data=(testX, testY), callbacks=[checkpointer],
		verbose=2, shuffle=True)
	
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	
	# invert predictions and targets to unscaled
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	
	# shift predictions of training data for plotting
	trainPredictPlot = np.empty_like(spy_prices)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift predictions of test data for plotting
	testPredictPlot = np.empty_like(spy_prices)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(spy_prices)-1, :] = testPredict
	
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(spy_prices))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()


def unit_test():
	run_test()
	
if __name__ =='__main__':
	unit_test()
