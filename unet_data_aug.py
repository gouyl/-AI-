import cv2
import random
import os
import numpy as np
from tqdm import tqdm_notebook

img_w = 256  
img_h = 256  

train_ids = next(os.walk("../data/jingwei_round1_train_20190619/256_train"))[2]

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.4:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 200000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(train_ids)
    g_count = 0
    for i,id_ in tqdm_notebook(enumerate(train_ids),total=len(train_ids)):
        count = 0
        src_img = cv2.imread('../data/jingwei_round1_train_20190619/256_train/' + id_)  # 3 channels
        label_img = cv2.imread('../data/jingwei_round1_train_20190619/256_label/' + id_,cv2.IMREAD_GRAYSCALE)  # single channel
        if len(np.flatnonzero(src_img))/(256*256*3) > 0.5:

            while count < image_each:
                #random_width = random.randint(0, X_width - img_w - 1)
                #random_height = random.randint(0, X_height - img_h - 1)
                #src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
                #label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
                if mode == 'augment':
                    src_roi,label_roi = data_augment(src_img,label_img)
            
                #cv2.imwrite(('./unet_train/visualize/%d.png' % g_count),visualize)
                cv2.imwrite(('unet_train/256_train_aug20w/%d.png' % g_count),src_roi)
                cv2.imwrite(('unet_train/256_label_aug20w/%d.png' % g_count),label_roi)
                count += 1 
                g_count += 1
                print(g_count)


            
    

if __name__=='__main__':  
    creat_dataset(mode='augment')