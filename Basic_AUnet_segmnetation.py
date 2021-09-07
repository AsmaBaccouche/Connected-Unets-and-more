# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:05:38 2021

@author:  Asma Baccouche
"""


from __future__ import print_function

import os, glob
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Lambda, Input, concatenate, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, UpSampling2D, Add, Activation
from keras.optimizers import Adam, Adadelta
from keras.layers.merge import add,multiply
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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

def sens(y_true, y_pred):
    num=K.sum(K.multiply(y_true, y_pred))
    denom=K.sum(y_true)
    if denom==0:
        return 1
    else:
        return  num/denom

def spec(y_true, y_pred):
    num=K.sum(K.multiply(y_true==0, y_pred==0))
    denom=K.sum(y_true==0)
    if denom==0:
        return 1
    else:
        return  num/denom

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))


def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.8 * K.pow((1-BCE_EXP), 2.) * BCE)
    return focal_loss

def seg_loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):
	shape_x = K.int_shape(x)  # 32
	shape_g = K.int_shape(g)  # 16
	theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
	shape_theta_x = K.int_shape(theta_x)
	phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
	upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
	concat_xg = add([upsample_g, theta_x])
	act_xg = Activation('relu')(concat_xg)
	psi = Conv2D(1, (1, 1), padding='same')(act_xg)
	sigmoid_xg = Activation('sigmoid')(psi)
	shape_sigmoid = K.int_shape(sigmoid_xg)
	upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
	upsample_psi = expend_as(upsample_psi, shape_x[3])
	y = multiply([upsample_psi, x])
	result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
	result_bn = BatchNormalization()(result) 
	return result_bn

def UnetGatingSignal(inputs, is_batchnorm=False):
    shape = K.int_shape(inputs)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(inputs)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def get_aunet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    gating = UnetGatingSignal(conv5, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 256)
    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv5), attn_1], axis=3)  
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    gating = UnetGatingSignal(conv6, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 128)
    up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv6), attn_2], axis=3) 
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    gating = UnetGatingSignal(conv7, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 64)
    up8 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv7), attn_3], axis=3) 
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    gating = UnetGatingSignal(conv8, is_batchnorm=True)  
    attn_4 = AttnGatingBlock(conv1, gating, 32)
    up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv8), attn_4], axis=3) 
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(1e-4), loss=[seg_loss], metrics=[dice_coef, iou_coef])
    return model

name = 'mydata'

imgs_train1, imgs_mask_train1 = load_train_data('inbreast_cycleGAN')
imgs_train2, imgs_mask_train2 = load_train_data('mydata')

imgs_train = np.concatenate([imgs_train1, imgs_train2])
imgs_mask_train = np.concatenate([imgs_mask_train1, imgs_mask_train2])

#imgs_train, imgs_mask_train = load_train_data(name)

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
model = get_aunet()

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
