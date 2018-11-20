import os
import os.path
import pandas as pd
import numpy as np
import cv2
import threading

from keras.applications.inception_v3 import InceptionV3 # import model
from keras.applications.inception_v3 import preprocess_input #
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling2D # import layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger # import cac function can thiet
from keras.utils.np_utils import to_categorical # giong label encoding?
from keras import *
from keras.models import *
from keras.preprocessing import image
from keras import backend as K
from random import randint
import random
from clr_callback import CyclicLR # improt cyclical learning rate

NUMBER_OF_CLASSES = 10
SIZE = 312 # size gi ??
NUMBER_OF_FOLD = 5 # chia de train tren cac model khac nhau
BATCH_SIZE = 48
EPOCH = 100

# ftype = 'mfcc'
ftype = 'melspectrogram' # chuyen doi data music thanh cac feature

class ThreadSafeIterator:
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return self.it.__next__()

def threadsafe_generator(f):
	"""
	A decorator that takes a generator function and makes it thread-safe.
	"""

	def g(*args, **kwargs):
		return ThreadSafeIterator(f(*args, **kwargs))

	return g

@threadsafe_generator
def train_generator(df, batch_size):
	while True:
		df = df.sample(frac=1, random_state = randint(11, 99)).reset_index(drop=True)
		for start in range(0, df.shape[0], batch_size):
			end = min(start + batch_size, df.shape[0])
			sub_df = df.iloc[start:end,:]
			x_batch = []
			y_batch = []
			for index, row in sub_df.iterrows():
				img_path = row[ftype]
				img = cv2.imread(img_path)
				img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
				
				if random.random() < 0.5:
					img = np.fliplr(img)
				
				img = image.img_to_array(img)
				img = preprocess_input(img)
				x_batch.append(img)
				y_batch.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
			yield np.array(x_batch), np.array(y_batch)

@threadsafe_generator
def valid_generator(df, batch_size):
	while True:
		for start in range(0, df.shape[0], batch_size):
			end = min(start + batch_size, df.shape[0])
			sub_df = df.iloc[start:end,:]
			x_batch = []
			y_batch = []
			for index, row in sub_df.iterrows():
				img_path = row[ftype]
				img = cv2.imread(img_path)
				img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
				img = image.img_to_array(img)
				img = preprocess_input(img)
				x_batch.append(img)
				y_batch.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
			yield np.array(x_batch), np.array(y_batch)

def InceptionV3_Model():
	input_tensor = Input(shape=(SIZE, SIZE, 3))
	base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
	bn = BatchNormalization()(input_tensor) # tao batch data input
	x = base_model(bn) # base mode with input data
	x = GlobalMaxPooling2D()(x) # output cua base model di qua lop max pooling
	x = Dense(256, activation='relu')(x) # output cua lop max pooling dua qua lop dense
	x = Dropout(0.25)(x)
	output_tensor = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
	model = Model(inputs=input_tensor, outputs=output_tensor)
	return model

if __name__ == '__main__':
	# for fold in range(NUMBER_OF_FOLD):
	for fold in range(0,2,1):
		print('***************  Fold %d  ***************'%(fold))
		train_df = pd.read_csv('../../data/train_set_fold%d.csv'%(fold))
		valid_df = pd.read_csv('../../data/valid_set_fold%d.csv'%(fold))
		train_size = train_df.shape[0]
		valid_size = valid_df.shape[0]
		train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
		valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
		print('TRAIN SIZE: %d VALID SIZE: %d'%(train_size, valid_size))

		WEIGHTS_BEST = '../weights/best_weights_fold%d_%s.hdf5'%(fold,ftype)
		TRAINING_LOG = '../logs/training_logs_fold%d_%s.csv'%(fold,ftype)
		early_stoping = EarlyStopping(monitor='val_acc', patience=8, verbose=1)
		save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True, mode='max')
		reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 4, min_lr = 1e-8, verbose=1)
		csv_logger = CSVLogger(TRAINING_LOG, append=True)
		clr = CyclicLR(base_lr=1e-8, max_lr=4e-5, step_size=2000., mode='exp_range', gamma=0.99994)

		callbacks_warmup = [save_checkpoint,csv_logger]
		callbacks_clr = [early_stoping, save_checkpoint, clr, csv_logger]
		callbacks = [early_stoping, save_checkpoint, reduce_lr, csv_logger]

		model = InceptionV3_Model()

		# warm up
		for layer in model.layers[0:-3]:
			layer.trainable = False
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=8e-5), metrics=['accuracy'])
		model.summary()
		model.fit_generator(generator=train_generator(train_df, BATCH_SIZE), steps_per_epoch=train_steps, epochs=1, verbose=1,
							validation_data=valid_generator(valid_df, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks_warmup)

		for layer in model.layers:
			layer.trainable = True
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=4e-5), metrics=['accuracy'])
		model.summary()
		model.fit_generator(generator=train_generator(train_df, BATCH_SIZE), steps_per_epoch=train_steps, epochs=EPOCH, verbose=1,
							validation_data=valid_generator(valid_df, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks)

	K.clear_session()
