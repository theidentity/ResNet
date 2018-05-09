import numpy as np

from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D,Input
# from keras.layers import Conv2D,MaxPool2D,UpSampling2D
# from keras.layers import Input,Dropout,merge,concatenate
from keras.models import Model,save_model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50

import datetime
import itertools
import cv2

class ImageClassifier(object):
	"""docstring for ImageClassifier"""
	def __init__(self):

		self.img_rows,self.img_cols = 197,197
		self.num_epochs = 15
		self.batch_size = 32
		self.seed = 42
		self.input_shape = (self.img_rows,self.img_cols,3)
		self.num_classes = 5
		# self.steps_per_epoch = 8751//self.batch_size,
		# self.validation_steps = 3753//self.batch_size,

		self.train_img_path = 'data/train/'
		self.validation_img_path = 'data/validation/'

		self.name = 'ResNet50'
		self.save_path = ''.join(['models/',self.name,'_best','.h5'])

		self.model = self.get_model()

	def get_model(self):

		base_model = ResNet50(include_top=False,input_shape=self.input_shape)

		x = base_model.output
		x = Flatten()(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		predictions = Dense(self.num_classes,activation='softmax')(x)
		
		model = Model(base_model.inputs,predictions)
		return model

	def build_model(self,lr=1e-4):
		
		opt = Adam(lr=lr)
		self.model.compile(
			optimizer = opt,
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)

	def get_train_generator(self,path):
		img_gen = ImageDataGenerator(
			zoom_range = 0.2,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			horizontal_flip = False,
			rotation_range = 10.0,
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb'
			)

		return img_gen

	def get_validation_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb'
			)

		return img_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		
		return [early_stopping,checkpointer]
		# return [early_stopping,checkpointer,tensorboard]
		# return [checkpointer]

	def train(self,lr=1e-4):

		self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = self.num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def continue_training(self,lr=1e-4,num_epochs=10):

		self.model = load_model(self.save_path)
		# self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)


		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def normalize_array(self,arr,lower=0,upper=255,mode='minmax'):
		arr = np.array(arr,dtype=np.float32)
		if mode == 'minmax':
			arr = (upper-lower)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
		else:
			arr = (arr-np.mean(arr))/np.std(arr)
		return arr

	def get_predictions(self):
		pass


if __name__ == '__main__':
	m1 = ImageClassifier()
	# m1.train(lr=1e-4)
	m1.continue_training(lr=1e-5,num_epochs=3)
