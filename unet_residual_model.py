import os
import sys
import random
import pandas as pd
import numpy as np

from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical  
from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time



filepath_train = "../data/jingwei_round1_train_20190619/256_train/"
filepath_label = "../data/jingwei_round1_train_20190619/256_label/"

save_model_name = "model_unet_resnet_2w.h5"
im_height = 256
im_width = 256
im_chan = 3
n_label = 4
batch_size = 4
epochs = 30

classes = [0, 1, 2, 3]
Labelencoder = LabelEncoder()
Labelencoder.fit(classes)

'''
train_ids = next(os.walk(dir_train))[2]
label_ids = next(os.walk(dir_label))[2]
print(len(train_ids))
print(len(label_ids))


X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height*im_width,4), dtype=np.uint8)

for n , id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
	path = dir_train
	img = load_img(path + id_)
	x = img_to_array(img)[:,:,:]
	X_train[n] = x
	label = img_to_array(load_img(dir_label+id_, grayscale=True)).reshape((im_height*im_width))
	label = Labelencoder.transform(label)
	label = to_categorical(label, num_classes=n_label)
	Y_train[n] = label.reshape((im_height*im_width,n_label))
	#print(Y_train[n].shape, Y_train[n])
print("It works")

FCN 、SegNet 、U-Net、Dilated Convolutions 、DeepLab (v1 & v2) 、RefineNet 、PSPNet 、Large Kernel Matters 、DeepLab v3

'''


def get_train_val(val_rate=0.2):
	train_url = []
	train_set = []
	val_set = []

	for pic in os.listdir(filepath_train):
		train_url.append(pic)
	random.shuffle(train_url)

	total_num = len(train_url)
	val_num = int(val_rate*total_num)
	for i in range(len(train_url)):
		if i < val_num:
			val_set.append(train_url[i])
		else:
			train_set.append(train_url[i])

	return train_set, val_set

def generateData(batch_size, data=[]):
	while True:
		train_data = []
		train_label = []
		batch = 0
		for i in range(len(data)):
			url =data[i]
			batch += 1 
			img = load_img(filepath_train + url)
			img = img_to_array(img)
			
			train_data.append(img)
			label = load_img(filepath_label + url, grayscale=True)
			label = img_to_array(label).reshape((im_width*im_height,))
			train_label.append(label)

			if batch % batch_size == 0:
				train_data = np.array(train_data)
				train_label = np.array(train_label).flatten()
				train_label = Labelencoder.transform(train_label)
				train_label = to_categorical(train_label, num_classes=n_label)
				train_label = train_label.reshape((batch_size, im_width * im_height, n_label))
				yield (train_data, train_label)
				train_data = []
				train_label =[]
				batch = 0

def generateValidData(batch_size, data=[]):
	while True:
		valid_data = []
		valid_label = []
		batch = 0
		for i in range(len(data)):
			url = data[i]
			batch += 1
			img = load_img(filepath_train + url)
			img = img_to_array(img)
			valid_data.append(img)
			label = load_img(filepath_label+url, grayscale=True)
			label=img_to_array(label).reshape((im_width*im_height,))
			valid_label.append(label)
			if batch % batch_size == 0:
				valid_data = np.array(valid_data)
				valid_label = np.array(valid_label).flatten()
				valid_label = Labelencoder.transform(valid_label)
				valid_label = to_categorical(valid_label, num_classes=n_label)
				valid_label = valid_label.reshape((batch_size, im_width*im_height, n_label))
				yield(valid_data, valid_label)
				valid_data= []
				valid_label = []
				batch = 0





def BatchActivate(x):
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
	x = Conv2D(filters, size, strides=strides, padding=padding)(x)
	if activation == True:
		x = BatchActivate(x)
	return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
	x = BatchActivate(blockInput)
	x = convolution_block(x, num_filters, (3,3) )
	x = convolution_block(x, num_filters, (3,3), activation=False)
	x = Add()([x, blockInput])
	if batch_activate:
		x = BatchActivate(x)
	return x



# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.4):
	# 101 -> 50
	s = Lambda(lambda x:x/255)(input_layer)
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(s)
	conv1 = residual_block(conv1,start_neurons * 1)
	conv1 = residual_block(conv1,start_neurons * 1, True)
	pool1 = MaxPooling2D((2, 2))(conv1)
	pool1 = Dropout(DropoutRatio/2)(pool1)

	# 50 -> 25
	conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
	conv2 = residual_block(conv2,start_neurons * 2)
	conv2 = residual_block(conv2,start_neurons * 2, True)
	pool2 = MaxPooling2D((2, 2))(conv2)
	pool2 = Dropout(DropoutRatio)(pool2)

	# 25 -> 12
	conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
	conv3 = residual_block(conv3,start_neurons * 4)
	conv3 = residual_block(conv3,start_neurons * 4, True)
	pool3 = MaxPooling2D((2, 2))(conv3)
	pool3 = Dropout(DropoutRatio)(pool3)

	# 12 -> 6
	conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
	conv4 = residual_block(conv4,start_neurons * 8)
	conv4 = residual_block(conv4,start_neurons * 8, True)
	pool4 = MaxPooling2D((2, 2))(conv4)
	pool4 = Dropout(DropoutRatio)(pool4)

	# Middle
	convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
	convm = residual_block(convm,start_neurons * 16)
	convm = residual_block(convm,start_neurons * 16, True)
	
	# 6 -> 12
	deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(DropoutRatio)(uconv4)
	
	uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
	uconv4 = residual_block(uconv4,start_neurons * 8)
	uconv4 = residual_block(uconv4,start_neurons * 8, True)
	
	# 12 -> 25
	deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
	#deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
	uconv3 = concatenate([deconv3, conv3])    
	uconv3 = Dropout(DropoutRatio)(uconv3)

	uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
	uconv3 = residual_block(uconv3,start_neurons * 4)
	uconv3 = residual_block(uconv3,start_neurons * 4, True)

	# 25 -> 50
	deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
	uconv2 = concatenate([deconv2, conv2])
		
	uconv2 = Dropout(DropoutRatio)(uconv2)
	uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
	uconv2 = residual_block(uconv2,start_neurons * 2)
	uconv2 = residual_block(uconv2,start_neurons * 2, True)
	
	# 50 -> 101
	deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	#deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
	uconv1 = concatenate([deconv1, conv1])

	uconv1 = Dropout(DropoutRatio)(uconv1)
	uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
	uconv1 = residual_block(uconv1,start_neurons * 1)
	uconv1 = residual_block(uconv1,start_neurons * 1, True)
	
	#uconv1 = Dropout(DropoutRatio/2)(uconv1)
	#output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
	output_layer_noActi = Conv2D(n_label, (1,1), padding="same", activation=None)(uconv1)
	output_layer =  Activation('softmax')(output_layer_noActi)
	output_layer = Reshape((im_height*im_width, n_label))(output_layer)
	
	return output_layer


def get_iou_vector(A, B):
	batch_size = A.shape[0]
	metric = []
	for batch in range(batch_size):
		t, p = A[batch]>0, B[batch]>0
		intersection = np.logical_and(t, p)
		union = np.logical_or(t, p)
		iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
		thresholds = np.arange(0.5, 1, 0.05)
		s = []
		for thresh in thresholds:
			s.append(iou > thresh)
		metric.append(np.mean(s))

	return np.mean(metric)

def my_iou_metric(label, pred):
	return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)





input_layer = Input((im_height, im_width, 3))
output_layer = build_model(input_layer, 16,0.4)

model = Model(input_layer, output_layer)

c = optimizers.adam(lr = 0.001)
model.compile(loss="categorical_crossentropy", optimizer=c, metrics=[my_iou_metric])

model.summary()

model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
									mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

train_set , val_set = get_train_val()
train_num = len(train_set)
valid_num = len(val_set)
print("the number of train data is:", train_num)
print("the number of val data is:", valid_num)

history = model.fit_generator(generator=generateData(batch_size,train_set),steps_per_epoch=train_num//batch_size,
	epochs=epochs,validation_data=generateValidData(batch_size, val_set), validation_steps=valid_num//batch_size,
	callbacks=[model_checkpoint,reduce_lr],verbose=1)

print("Train Done")