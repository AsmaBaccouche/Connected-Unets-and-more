# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:06:54 2021

@author: Asma Baccouche
"""

from __future__ import print_function

import os, glob
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Dense, Reshape, GlobalAveragePooling2D, Multiply, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from data import load_train_data, load_test_data
from sklearn.model_selection import train_test_split
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


img_rows = 256
img_cols = 256
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.8 * K.pow((1-BCE_EXP), 2.) * BCE)
    return focal_loss

def loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def resnet_block(x, n_filter, strides=1):
    x_init = x
    ## Conv 1
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)
    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def get_runet():    
    inputs = Input((img_rows, img_cols, 3))
    
    conv1 = resnet_block(inputs,32 , strides=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = resnet_block(pool1,64 , strides=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = resnet_block(pool2, 128, strides=1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = resnet_block(pool3, 256, strides=1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = resnet_block(up6, 256, strides=1)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = resnet_block(up7, 128, strides=1)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = resnet_block(up8, 64, strides=1)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = resnet_block(up9, 32, strides=1)
            
    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv9])
    model.compile(optimizer=Adam(), loss=[loss], metrics=[dice_coef, iou_coef])
    return model


imgs_train1, imgs_mask_train1 = load_train_data('inbreast_cycleGAN')
imgs_train2, imgs_mask_train2 = load_train_data('mydata')

imgs_train = np.concatenate([imgs_train1, imgs_train2])
imgs_mask_train = np.concatenate([imgs_mask_train1, imgs_mask_train2])

#imgs_train, imgs_mask_train = load_train_data(name)

name = 'join_inbreast6'

fname = 'basic_aunet_'+name+'_weights.h5'
pred_dir = fname[:-11]

imgs_train = imgs_train.astype('float32')

mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std
imgs_mask_train = imgs_mask_train.astype('float32')

imgs_mask_train /= 255.  # scale masks to [0, 1]
imgs_mask_train = imgs_mask_train[..., np.newaxis]

imgs_train, imgs_val, imgs_mask_train, imgs_mask_val = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=42)
print('-'*30)
print('Creating and compiling model...')
print('-'*30)
model = get_runet()

model_checkpoint = ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)

history = model.fit(imgs_train, imgs_mask_train,
                    batch_size=8, epochs=100, verbose=1, shuffle=True,
                    validation_data=(imgs_val, imgs_mask_val),
                    callbacks=[model_checkpoint])

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)

imgs_test1, imgs_id_test1 = load_test_data('inbreast_cycleGAN')
imgs_test2, imgs_id_test2 = load_test_data('mydata')

imgs_test = np.concatenate([imgs_test1, imgs_test2])
imgs_id_test = np.concatenate([imgs_id_test1, imgs_id_test2])

#imgs_test, imgs_id_test = load_test_data(name)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights(fname)

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('imgs_mask_test_'+name+'.npy', imgs_mask_test)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)

if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
    
#data_path2 = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Test_Seg/'
data_path2 = 'D:/INbreast/Test_Seg/'
#data_path2 = 'D:/CBIS_augmented/Test_Seg/'
#data_path2 = 'D:/CSAW-S/CsawS/Test_Seg/'

d = data_path2+'roi/*.png'    
files = glob.glob(d) 

files1 = files

data_path2 = 'D:/Files/MYDATA/Breast_Cancer-Begonya/Images/Test_Seg/'

d = data_path2+'roi/*.png'    
files = glob.glob(d) 

files2 = files

files = files1 + files2


files = [file.split('\\')[-1][:-4] for file in files]
idx = 0
for image, image_id in zip(imgs_mask_test, imgs_id_test):
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, files[idx] + '_pred.png'), image)
    idx = idx + 1

imgs_id_test = imgs_id_test.astype('float32')
imgs_id_test = imgs_id_test[..., np.newaxis]
imgs_id_test = imgs_id_test // 255

ev = model.evaluate(imgs_test, imgs_id_test)
dice, iou = ev[1], ev[2]

print("dice score:", dice)
print("iou score:", iou)

   
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice coef')
plt.ylabel('dice coef')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['iou_coef'])
plt.plot(history.history['val_iou_coef'])
plt.title('model iou coef')
plt.ylabel('iou coef')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
