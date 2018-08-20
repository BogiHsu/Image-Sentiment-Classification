from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Flatten, Activation
dr = 0.5
nb_class = 7

def build_model(mode):
	if mode == 'dnn':
		return dnn_model()
	if mode == 'cnn':
		return cnn_model()

def cnn_model():
	model = Sequential()
	
	# CNN
	# 1
	model.add(Conv2D(128, 3, padding = 'same', activation = 'relu', input_shape = (48, 48, 1)))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(dr))
	# 2
	model.add(Conv2D(256, 3, padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(dr))
	# 3
	model.add(Conv2D(256, 3, padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(dr))
	# 4
	model.add(Conv2D(128, 3, padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(dr))

	# DNN
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
	
	return model

def dnn_model():
	model = Sequential()
	
	# DNN
	model.add(Flatten(input_shape = (48, 48, 1)))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
	
	return model
