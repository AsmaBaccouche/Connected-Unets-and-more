# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:53:51 2020

@author:  Asma Baccouche
"""

from __future__ import print_function
import os
import numpy as np
import cv2

image_rows = 256
image_cols = 256

def create_train_data(data_path, name):
    train_data_path = os.path.join(data_path, 'img')
    train_mask_path = os.path.join(data_path, 'msk')    
    images = os.listdir(train_data_path)
    masks = os.listdir(train_mask_path)
    total = len(images)
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for j in range(len(images)):

        img = cv2.imread(os.path.join(train_data_path, images[j]))
        img = cv2.resize(img, (256,256))
        #enhancement
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img_eqhist=cv2.equalizeHist(gray_img)
        img = cv2.cvtColor(gray_img_eqhist, cv2.COLOR_GRAY2BGR)
        #end enhancement
        img_mask = cv2.imread(os.path.join(train_mask_path, masks[j]), 0)
        img_mask = cv2.resize(img_mask, (256,256))
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train_'+name+'.npy', imgs)
    np.save('imgs_mask_train_'+name+'.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(name):
    imgs_train = np.load('UNets files/imgs_train_'+name+'.npy')
    imgs_mask_train = np.load('UNets files/imgs_mask_train_'+name+'.npy')
    return imgs_train, imgs_mask_train


def create_test_data(data_path2, name):
    test_data_path = os.path.join(data_path2, 'img')
    test_mask_path = os.path.join(data_path2, 'msk')    
    images = os.listdir(test_data_path)
    masks = os.listdir(test_mask_path)
    total = len(images)
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating testing images...')
    print('-'*30)
    for j in range(len(images)):

        img = cv2.imread(os.path.join(test_data_path, images[j]))
        img = cv2.resize(img, (256,256))
        #enhancement
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img_eqhist=cv2.equalizeHist(gray_img)
        img = cv2.cvtColor(gray_img_eqhist, cv2.COLOR_GRAY2BGR)
        #end enhancement
        img_mask = cv2.imread(os.path.join(test_mask_path, masks[j]), 0)
        img_mask = cv2.resize(img_mask, (256,256))
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')


    np.save('imgs_test_'+name+'.npy', imgs)
    np.save('imgs_id_test_'+name+'.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_test_data(name):
    imgs_test = np.load('UNets files/imgs_test_'+name+'.npy')
    imgs_id = np.load('UNets files/imgs_id_test_'+name+'.npy')
    return imgs_test, imgs_id

#data_path = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Train_Seg/'
#data_path2 = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Test_Seg/'

#data_path = 'D:/CBIS_augmented/Train_Seg/'
#data_path2 = 'D:/CBIS_augmented/Test_Seg/'
#
##data_path = 'D:/INbreast/Train_Seg/'
##data_path2 = 'D:/INbreast/Test_Seg/'
#
##data_path = 'D:/CSAW-S/CsawS/Train_Seg/'
##data_path2 = 'D:/CSAW-S/CsawS/Test_Seg/'
#
#name = 'CBIS_cycleGAN'
#
    
#name = 'entire_mydata'
#
#data_path = 'D:/Files/MYDATA/train/'
#data_path2 = 'D:/Files/MYDATA/test/'
#
#create_train_data(data_path, name)
#create_test_data(data_path2, name)
