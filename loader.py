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
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from interest_rate_loader import InterestRateLoader

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
		return pd.to_datetime(x, infer_datetime_format=True)
		
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
		
		# interest rate loader
		iloader = InterestRateLoader()
		data_int_rate = iloader.load_all_data(all=True)

		# merge interest rates
		data_merged = data_spy.merge(data_int_rate, how='outer', left_index=True, right_index=True)
		data_merged.to_csv("data/merged_irate.csv")
		
		#clean up data by fill foward
		data_cleaned = data_merged.fillna(method='ffill')
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
		#supervised.drop(supervised.columns[[10,11,12,13,15,16,17,18,19]], axis=1, inplace=True)
		print(supervised.head())

		# split into train and test sets
		values = supervised.values
		n_train_days = 252 * 15 #  252 trading days per year muliplied by X years
		train = values[:n_train_days, :]
		test = values[n_train_days:, :]	
		print("start")
		print(train.shape, test.shape)
		# split into input and outputs
		X_train, y_train = train[:, :-1], train[:, -1]
		X_test, y_test = test[:, :-1], test[:, -1]
		print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
		# reshape input to be 3D [samples, timesteps, features]
		X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
		X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
		print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
		
		# design network
		model = Sequential()
		model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='sigmoid'))
		model.add(Dropout(0.2))
		model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False, activation='sigmoid'))
		model.add(Dropout(0.2))
		model.add(Dense(32,kernel_initializer="uniform", activation='relu'))        
		model.add(Dense(1,kernel_initializer="uniform", activation='linear'))		
		model.summary()
		
		# compile model
		model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

		# train the model; save model file when improved
		checkpointer = ModelCheckpoint(filepath='saved_models/money.hdf5', verbose=1, save_best_only=True)
		history = model.fit(X_train, y_train, epochs=50, batch_size=80,
			validation_data=(X_test, y_test), 
			verbose=2, shuffle=True)		
		#callbacks=[checkpointer],
		
		# plot history
		#pyplot.plot(history.history['loss'], label='train')
		#pyplot.plot(history.history['val_loss'], label='test')
		#pyplot.legend()
		#pyplot.show()
		
		# make a prediction
		yhat = model.predict(X_test)
		X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
		# invert scaling for forecast
		inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)
		inv_yhat = inv_yhat[:,0]
		# invert scaling for actual
		y_test = y_test.reshape((len(y_test), 1))
		inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[:,0]
		print("predicted price of SPY: ", inv_y[-1])
		# calculate RMSE
		rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
		print('Test RMSE: %.3f' % rmse)
		
def unit_test():
	loader = DataLoader()
	loader.load_all_data()
	
if __name__ =='__main__':
	unit_test()