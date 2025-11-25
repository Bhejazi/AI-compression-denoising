#%% Import libraries
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:23:37 2024

@author: bhejazi
"""
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import tensorflow.keras.callbacks
from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot
from PIL import Image
from colorama import Fore, Style
# import json

#%% Preparing Dataset
# Remembr to change dtype depending on image format
 
def load_images(path):
    folder_info = os.listdir(path)
    print("Loading images..")
        
    file_idx = [i for i, file_name in enumerate(folder_info) if file_name[-4:] == "tiff"]
    #file_idx = [i for i, file_name in enumerate(folder_info)]
    
    n_images = len(file_idx)
    temp_img = np.array(Image.open(os.path.join(path, folder_info[file_idx[0]])))
    rows, cols = temp_img.shape
    
    images = np.zeros((n_images, rows, cols), dtype="uint8")

    for i, idx in enumerate(file_idx):
        images[i, :, :] = np.array(Image.open(os.path.join(path, folder_info[idx])))
      
    return images
       
while True:
    print("\n" + "Enter image file path:" + Fore.BLUE + Style.BRIGHT)
    path = input ("-> ")
    print(Style.RESET_ALL)

    if os.path.exists(path): #check if path is valid
        break
    else:
        print (Fore.RED + Style.BRIGHT + "Path does not exist, re-enter correct path")

images = load_images(path)

x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

# normalize the image data
if np.max(x_train) > 255:
    x_train = x_train.astype('float32') / 65535
    x_test = x_test.astype('float32') / 65535
elif np.max(x_train) > 1 & np.max(x_train) < 255:
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

img_size = images.shape[1]
# reshape in the input data for the model
x_train = x_train.reshape(len(x_train), img_size, img_size, 1)
x_test = x_test.reshape(len(x_test), img_size, img_size, 1)

#%% Encoder

# !!! Make sure to adjust the layers properly with mirror number of covolutions in layers !!!
# edit number of layers for different compression rates

x = tensorflow.keras.layers.Input(shape=(img_size,img_size,1), name="encoder_input")

encoder_Conv2D_1 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
encoder_MaxPooling_1 = tensorflow.keras.layers.MaxPooling2D(2, padding='same')(encoder_Conv2D_1)

encoder_Conv2D_2 = tensorflow.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(encoder_MaxPooling_1)
encoder_MaxPooling_2 = tensorflow.keras.layers.MaxPooling2D(2, padding='same')(encoder_Conv2D_2)

encoder_Conv2D_3 = tensorflow.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(encoder_MaxPooling_2)
#encoder_MaxPooling_3 = tensorflow.keras.layers.MaxPooling2D(2, padding='same')(encoder_Conv2D_3)

#encoder_Conv2D_4 = tensorflow.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(encoder_MaxPooling_3)
#encoder_MaxPooling_4 = tensorflow.keras.layers.MaxPooling2D(2, padding='same')(encoder_Conv2D_4)

#encoder_Conv2D_5 = tensorflow.keras.layers.Conv2D(2, 3, activation='relu', padding='same')(encoder_MaxPooling_4)

encoder_output = encoder_Conv2D_3

encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")
encoder.summary()
#%% Decoder
decoder_input = tensorflow.keras.layers.Input(shape=(encoder_output.shape[1],encoder_output.shape[1], encoder_output.shape[3]), name="decoder_input")

#decoder_Conv2D_1 = tensorflow.keras.layers.Conv2D(2, 3, activation='relu', padding='same')(decoder_input)
#decoder_UpSampling2D_1 = tensorflow.keras.layers.UpSampling2D(2)(decoder_Conv2D_1)

#decoder_Conv2D_1 = tensorflow.keras.layers.Conv2D(4, 3, activation='relu', padding='same')(decoder_input)
#decoder_UpSampling2D_1 = tensorflow.keras.layers.UpSampling2D(2)(decoder_Conv2D_1)

decoder_Conv2D_1 = tensorflow.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(decoder_input)
decoder_UpSampling2D_1 = tensorflow.keras.layers.UpSampling2D(2)(decoder_Conv2D_1)

decoder_Conv2D_2 = tensorflow.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(decoder_UpSampling2D_1)
decoder_UpSampling2D_2 = tensorflow.keras.layers.UpSampling2D(2)(decoder_Conv2D_2)

decoder_Conv2D_3 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(decoder_UpSampling2D_2)

decoder_output = tensorflow.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(decoder_Conv2D_3)

decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()
#%% Autoencoder
ae_input = tensorflow.keras.layers.Input(shape=(img_size,img_size,1), name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = tensorflow.keras.models.Model(ae_input, ae_decoder_output, name="AE")
ae.summary()

# AE Compilation
ae.compile(optimizer='adam', loss='binary_crossentropy')
#%% Training AE
# Remembr to change training log file name
#csv_logger = tensorflow.keras.callbacks.CSVLogger('D:/bhejazi/compression/models/metrics/8bit-orig-images_training-log.csv', append=False)
#ae.fit(x_train, x_train, epochs=100, batch_size=128, validation_data=(x_test, x_test), callbacks=[csv_logger])
ae.fit(x_train, x_train, epochs=100, batch_size=128, validation_data=(x_test, x_test))
#%% Apply model to images
encoded_images = encoder.predict(x_train)
decoded_images = decoder.predict(encoded_images)
#%% Display images
num_images_to_show = 5
for im_ind in range(num_images_to_show):
    plot_ind = im_ind*3 + 1
    rand_ind = np.random.randint(low=0, high=x_train.shape[0])
    
    matplotlib.pyplot.subplot(num_images_to_show, 3, plot_ind)
    matplotlib.pyplot.imshow(x_train[rand_ind, :, :], cmap="gray")
      
    matplotlib.pyplot.subplot(num_images_to_show, 3, plot_ind+1)
    matplotlib.pyplot.imshow(encoded_images[rand_ind, :, :, 1], cmap="gray")
    
    matplotlib.pyplot.subplot(num_images_to_show, 3, plot_ind+2)
    matplotlib.pyplot.imshow(decoded_images[rand_ind, :, :], cmap="gray")
#%%
# matplotlib.pyplot.imshow(x_train[1, :, :], cmap="gray")
# matplotlib.pyplot.imshow(decoded_images[1, :, :], cmap="gray")
#%% Save images
size_conversion = "512x512-to-128x128x8"
file_extension = "binary-images-CNN"

encoded_path = f"M:/codes/codes_BAM/python/autoencoder/temp/encoded/encoded_compress_{size_conversion}-{file_extension}.npy"
decoded_path = f"M:/codes/codes_BAM/python/autoencoder/temp/decoded/decoded_compress_{size_conversion}-{file_extension}.npy"
train_images_path = f"M:/codes/codes_BAM/python/autoencoder/temp/train_images_compress-{file_extension}.npy"

np.save(encoded_path, encoded_images)
np.save(decoded_path, decoded_images)
# np.save(train_images_path, x_train)
#%%
#!mkdir -p saved_model
#model.save('saved_model/my_model')
#ae.save('M:/codes/codes_BAM/python/autoencoder/temp/model_compress_512x512-to-64x64x4-loss0.067-v2.keras') 

#%% Save model weights
encoder.save('M:/codes/codes_BAM/python/autoencoder/temp/models/encoder_model-512x512-to-64x64x4-8bit-orig-tifs.h5')
decoder.save('M:/codes/codes_BAM/python/autoencoder/temp/models/decoder_model-512x512-to-64x64x4-8bit-orig-tifs.h5')

#%% Model visualization
import visualkeras
visualkeras.layered_view(encoder, draw_volume=True, spacing=100, scale_xy=1, scale_z=1)
visualkeras.layered_view(decoder, draw_volume=True, spacing=100, scale_xy=1, scale_z=1, draw_reversed=True)
