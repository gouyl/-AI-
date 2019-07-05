import numpy as np  
from keras.models import Sequential,load_model,Input,Model
from keras.layers import Conv2D,MaxPooling2D,Dropout,UpSampling2D,BatchNormalization,Reshape,Permute,Activation  
from keras.layers.merge import concatenate
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array , load_img, ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
from keras.layers.core import Lambda
import random
import os, time
from tqdm import tqdm_notebook  
from sklearn.preprocessing import LabelEncoder  

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
dir_train = "data/jingwei_round1_train_20190619/256_train/"
dir_label = "data/jingwei_round1_train_20190619/256_label/"

im_height = 256
im_width = 256
im_chan = 3
n_label = 4
classes = [0,1,2,3]

Labelencoder = LabelEncoder()
Labelencoder.fit(classes)

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
	Y_train[n] = label.reshape((1,im_height*im_width,n_label))
	#print(Y_train[n].shape, Y_train[n])
print("It works")




def unet():
	inputs = Input((im_width, im_height,3))
	s = Lambda(lambda x:x/255)(inputs)
	conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(s)
	conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
	conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
	conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
	conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
	drop4 = Dropout(0.4)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
	conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([up6, conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
	conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
	up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
	conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

	up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
	conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

	up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
	conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

	#conv10 = Conv2D(n_label, (1, 1), strides=(1, 1), padding='same')(conv9)
	conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)
	conv10 = Reshape((im_width*im_height,n_label))(conv10)
	model = Model(inputs=inputs, outputs=conv10)
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model



t1 = time.time()

Earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint("model_unet_2w_simple.h5",verbose=1, save_best_only=True)

model = unet()
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=12, epochs=40, callbacks=[Earlystopper, checkpointer])

t2 = time.time()

print("Total time: %f" %(t2-t1))