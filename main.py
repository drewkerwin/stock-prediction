import argparse
import time
import numpy as np



if __name__ =='__main__':
	# args required file name
	parser = argparse.ArgumentParser(description='Evaluate image classification')
	parser.add_argument('--pklfile', required=False, help='pickle file name with list of image files')
	parser.add_argument('--train_perc', required=False, help='percentage (float e.g. 0.70) of samples that are used for training')
	parser.add_argument('--test_perc', required=False, help='percentage (float e.g. 0.15) of samples that are used for testing')
	parser.add_argument('--validation_perc', required=False, help='percentage (float e.g. 0.15) of samples for validation')
	parser.add_argument('--epochs', required=False, help='number of epochs to train')
	parser.add_argument('--model', required=False, help='model to evaluate data')
	parser.add_argument('--testonly', required=False, default='False', help='should trial perform test only on existing model file')
	parser.add_argument('--loss', required=False, default='categorical_crossentropy', help='type of loss function - https://keras.io/losses/#available-loss-functions')
	parser.add_argument('--optimizer', required=False, default='rmsprop', help='type of optimizer - https://keras.io/optimizers/')
	parser.add_argument('--metrics', required=False, default='accuracy', help='type of metrics - accuracy')
	parser.add_argument('--weightfile', required=False, default='saved_models/aug_model.weights.best.hdf5', help='file to read/write model')
	args = parser.parse_args()

	start = time.time()	
	

	endtrain = time.time()

	endtest = time.time()
	
	elapsedtrain = endtrain - start
	elapsedtest = endtest - endtrain
	et = '{}: elapsedtrain\n'.format(elapsedtrain)
	ett = '{}: elapsedtest\n'.format(elapsedtest)

	print(et)
	print(ett)
