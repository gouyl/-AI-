from keras.models import load_model
import numpy as np 
import os, time, random, re
from PIL import Image
from tqdm import tqdm_notebook
from keras.preprocessing.image import img_to_array, load_img
from scipy import misc

model = load_model("model_unet_2w.h5")

im_height = 256
im_width = 256
im_chan = 3

dir_test_3 = "../data/jingwei_round1_test_a_20190619/256_test_3/"
dir_test_4 = "../data/jingwei_round1_test_a_20190619/256_test_4/"
ids_test_3 = next(os.walk(dir_test_3))[2]
ids_test_4 = next(os.walk(dir_test_4))[2]
print(len(ids_test_3))
print(len(ids_test_4))
X_test_3 = np.zeros((len(ids_test_3), im_height, im_width, im_chan), dtype=np.uint8)
X_test_4 = np.zeros((len(ids_test_4), im_height, im_width, im_chan), dtype=np.uint8) 


for n, id_ in tqdm_notebook(enumerate(ids_test_3), total=len(ids_test_3)):
	img = load_img(dir_test_3+id_)
	x = img_to_array(img)[:,:,:]
	X_test_3[n] = x
print("Done,X_test_3")

for n, id_ in tqdm_notebook(enumerate(ids_test_4), total=len(ids_test_4)):
	img = load_img(dir_test_4+id_)
	x = img_to_array(img)[:,:,:]
	X_test_4[n] = x
print("Done,x_test_4")



pred_test_3 = model.predict(X_test_3, verbose=2)
pred_test_3 = np.argmax(pred_test_3, axis=2)
num, w_h = pred_tes_3.shape
print(num)
pred_test_3 = pred_test_3.reshape((num,256,256)).astype(np.uint8)
print(np.unique(pred_test_3))
zero_ratio = len(pred_test_3[pred_test_3==0])/(num*w_h)
one_ratio = len(pred_test_3[pred_test_3==1])/(num*w_h)
two_ratio = len(pred_test_3[pred_test_3==2])/(num*w_h)
print(zero_ratio, one_ratio, two_ratio)


pred_test_4 = model.predict(X_test_4, verbose=2)
pred_test_4 = np.argmax(pred_test_4, axis=2)
num, w_h = pred_test_4.shape
pred_test_4 = pred_test_4.reshape((num, 256,256)).astype(np.uint8)
print(np.unique(pred_test_4))


img_3_width = 37241
img_3_height  = 19903
number_row = int(img_3_height/256)
number_col = int(img_3_width/256)

Label_3_img = np.zeros((img_3_height, img_3_width), dtype=np.uint8) 

for n, id_ in tqdm_notebook(enumerate(ids_test_3), total=len(ids_test_3)):
		num = re.findall(r"\d+", id_)
		num = int(num[0])

		row = int(num / number_col)
		col = int(num % number_col)
		img = pred_test_3[n]
		Label_3_img[256*row:256*(row+1), 256*col:256*(col+1)] = img 

print(np.unique(Label_3_img))
misc.imsave("image_3_predict.png", Label_3_img)
print("Done: image 3")


img_4_height = 28832
img_4_width  = 25936
number_row = int(img_4_height/256)
number_col = int(img_4_width/256)

Label_4_img = np.zeros((img_4_height, img_4_width), dtype=np.uint8) 
 
for n, id_ in tqdm_notebook(enumerate(ids_test_4), total=len(ids_test_3)):
        num = re.findall(r"\d+", id_)
        num = int(num[0])
        row = int(num / number_col)
        col = int(num % number_col)
        img = pred_test_4[n]
        #img = np.squeeze(img)
        #img = np.rint(img).astype(np.uint8)
        Label_4_img[256*row:256*(row+1), 256*col:256*(col+1)] = img 

print(np.unique(Label_4_img))
misc.imsave("image_4_predict.png", Label_4_img)
print("Done: image 4")