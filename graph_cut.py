import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

Image.MAX_IMAGE_PIXELS = 10000000000
print("ok")
img_1 =Image.open('data/jingwei_round1_train_20190619/jingwei_round1_train_20190619/image_2.png')
img_2 = Image.open("data/jingwei_round1_train_20190619/jingwei_round1_train_20190619/image_2.png")

img_label_1 = Image.open("data/jingwei_round1_train_20190619/jingwei_round1_train_20190619/image_1_label.png")
img_label_2 = Image.open("data/jingwei_round1_train_20190619/jingwei_round1_train_20190619/image_2_label.png")

img_1_width, img_1_height = img_1.size
img_width_label, img_height_label = img_label_1

img_test_3 =Image.open('data/jingwei_round1_test_a_20190619/jingwei_round1_test_a_20190619/image_3.png')
img_test_4 =Image.open('data/jingwei_round1_test_a_20190619/jingwei_round1_test_a_20190619/image_4.png')

img_test_3_width, img_test_3_height =img_test_3.size
img_test_4_width, img_test_4_height =img_test_4.size

print("image_1: %d %d" %(img_1_width, img_1_height))
print("img_2: %d %d" % (img_2_width, img_2_height))
print("img_3: %d %d" %( img_test_3_width, img_test_3_height))
print("img_4: %d %d", img_test_4_width, img_test_4_height)


def cut_train(img_1,img_label_1, vx, vy):
	name_train = "data/jingwei_round1_train_20190619/32_train/graph"
	name_label = "data/jingwei_round1_train_20190619/32_label/graph"
	n = 0
	x1 = 0
	y1 = 0
	x2 = vx
	y2 = vy
	while x2 <= img_1_height:
		while y2 <= img_1_width:
			name_1 = name_train + str(n) + ".png" 
			name_1_label = name_label + str(n) + ".png"

			im_1 =img_1.crop((y1,x1,y2,x2))
			img_arr = np.array(im_1)
			if (len(np.flatnonzero(img_arr))/(32*32*3)) > 0.5:
				im_1.save(name_1)

				n = n + 1 
			y1 = y1 + vy
			y2 = y1 + vy
		x1 = x1 + vx
		x2 = x1 + vx
		y1 = 0
		y2 = vy

	print("number of graph:")
	return n

"""
	x1 = 0
	y1 = 0 
	x2 = vx
	y2 = vy
	while x2 <= img_2_height:
		while y2 <= img_2_width:
			name_2 = name_train + str(n) + ".png" 
			im_2 =img_2.crop((y1,x1,y2,x2))
			im_2.save(name_2)

			#name_2_label = name_label + str(n) + ".png" 
			#im_2_label = image_label_2.crop((y1,x1,y2,x2))
			#im_2_label.save(name_2_label)
			n = n + 1 
			y1 = y1 + vy
			y2 = y1 + vy
		x1 = x1 + vx
		x2 = x1 + vx
		y1 = 0
		y2 = vy
"""
	
def cut_test(img, vx, vy):
	name_test_3 = "data/jingwei_round1_test_a_20190619/256_test_3/graph"
	name_test_4 = "data/jingwei_round1_test_a_20190619/256_test_4/graph"
	n = 0 
	x1 = 0
	y1 = 0 
	x2 = vx
	y2 = vy
	while x2 <= img_test_4_height:
		while y2 <= img_test_4_width:
			name2 = name_test_4 + str(n) + ".png" 
			im2 =img.crop((y1,x1,y2,x2))
			im2.save(name2)
			n = n + 1 
			y1 = y1 + vy
			y2 = y1 + vy
		x1 = x1 + vx
		x2 = x1 + vx
		y1 = 0
		y2 = vy

	return n
'''
	x1 = 0
	y1 = 0 
	x2 = vx
	y2 = vy
	while x2 <= img_test_4_height:
		while y2 <= img_test_4_width:
			name2 = name_test_4 + str(n) + ".png" 
			im2 =img.crop((y1,x1,y2,x2))
			im2.save(name2)
			n = n + 1 
			y1 = y1 + vy
			y2 = y1 + vy
		x1 = x1 + vx
		x2 = x1 + vx
		y1 = 0
		y2 = vy	

	print("number of graph:")
	return n
'''
res1 = cut_train(img_1, 32, 32)
#res2 = cut_test(img_test_3, img_test_4, 256, 256)


print(res1)
