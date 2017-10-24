from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

class DataLoader:
	def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg
	
	def parse(self, x):
		return pd.to_datetime(x)
		
	def load_all_data(self):

		# load spy data with column names that will be unique later
		data_spy = read_csv('data/spy.csv', date_parser=self.parse, index_col=0, na_values=['nan'])
		data_spy.columns = ['SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 'SPY_CLOSE', 'SPY_ADJ_CLOSE', 'SPY_VOLUME']
		
		# load current vix csv, but disregard the disclaimer line
		data_vix_current = read_csv('data/vixcurrent.csv', date_parser=self.parse, index_col=0, skiprows=[0], na_values=['nan'])
		data_vix_current.columns = ['VIX_OPEN', 'VIX_HIGH', 'VIX_LOW', 'VIX_CLOSE']
		# ditto for archive
		data_vix_archive = read_csv('data/vixarchive.csv', date_parser=self.parse, index_col=0, skiprows=[0], na_values=['nan'])
		data_vix_archive.columns = ['VIX_OPEN', 'VIX_HIGH', 'VIX_LOW', 'VIX_CLOSE']
		
		# merge vix data - the data is available in different
		# time ranges, so merging together
		frames = [data_vix_archive, data_vix_current]
		data_vix = pd.concat(frames)
		
		# drop old data, save to file. 1995 is as far back as the SPY data goes.
		old_data = data_vix[(data_vix.index < '1995-01-01')].index
		data_vix = data_vix.drop(old_data)		
		data_vix.to_csv("data/vix.csv")

		#values = data_spy.values
		# specify columns to plot
		#groups = [0, 1, 2, 3, 5]
		#i = 1
		# plot each column
		#pyplot.figure()
		#for group in groups:
		#	pyplot.subplot(len(groups), 1, i)
		#	pyplot.plot(values[:, group])
		#	pyplot.title(data_spy.columns[group], y=0.5, loc='right')
		#	i += 1
		#pyplot.show()

		# merge spy & vix data
		data_merged = data_spy.merge(data_vix, how='outer', left_index=True, right_index=True)
		data_merged.to_csv("data/merged.csv")

		#clean up data by removing "NA" or blank data
		data_cleaned = data_merged.dropna()
		data_cleaned.to_csv("data/cleaned.csv")
		print(data_cleaned.head(5))
		
		# get values for LSTM processing
		values = data_cleaned.values
		values = values.astype('float32')

		# normalize features
		scaler = MinMaxScaler(feature_range=(0, 1))
		scaled = scaler.fit_transform(values)
		
		# frame as supervised learning
		supervised = self.series_to_supervised(scaled, 1, 1)
		# drop columns that are not predicted
		supervised.drop(supervised.columns[[10,11,12,13,15,16,17,18,19]], axis=1, inplace=True)
		print(supervised.head())

		# split into train and test sets
		values = supervised.values
		n_train_days = 252 * 10 #  252 trading days per year muliplied by X years
		train = values[:n_train_days, :]
		test = values[n_train_days:, :]
		# split into input and outputs
		train_X, train_y = train[:, :-1], train[:, -1]
		test_X, test_y = test[:, :-1], test[:, -1]
		# reshape input to be 3D [samples, timesteps, features]
		train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
		test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
		#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
		
		# design network
		model = Sequential()
		model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
		model.add(Dense(1))
		model.summary()
		
		# compile model
		model.compile(loss='mae', optimizer='adam')

		# train the model; save model file when improved
		checkpointer = ModelCheckpoint(filepath='saved_models/money.hdf5', verbose=1, save_best_only=True)
		history = model.fit(train_X, train_y, epochs=100, batch_size=100,
			validation_data=(test_X, test_y), callbacks=[checkpointer],
			verbose=2, shuffle=True)		
		
		# plot history
		pyplot.plot(history.history['loss'], label='train')
		pyplot.plot(history.history['val_loss'], label='test')
		pyplot.legend()
		pyplot.show()
		
		# make a prediction
		yhat = model.predict(test_X)
		print(yhat.shape)
		print(yhat)
		test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
		# invert scaling for forecast
		inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)
		inv_yhat = inv_yhat[:,0]
		print(inv_yhat.shape)
		print(inv_yhat)
		# invert scaling for actual
		test_y = test_y.reshape((len(test_y), 1))
		inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[:,0]
		print(inv_y.shape)
		print(inv_y)
		# calculate RMSE
		rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
		print('Test RMSE: %.3f' % rmse)
		
def unit_test():
	loader = DataLoader()
	loader.load_all_data()
	
if __name__ =='__main__':
	unit_test()