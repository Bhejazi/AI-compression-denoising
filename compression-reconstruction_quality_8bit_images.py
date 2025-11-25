# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:30:41 2025
---
Evaluate compressed and reconstructed image quality as compared to original 8 bit images

Side by side comparison and thresholded comparison
---
@author: bhejazi
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#%% Load output images from compression codes
size_conversion = "512x512-to-128x128x8"
file_extension = "8bit-orig-tifs-VQ-VAE"

orig_img_path = f"M:/codes/codes_BAM/python/autoencoder/temp/train_images_compress-{file_extension}.npy"
compressed_img_path = f"M:/codes/codes_BAM/python/autoencoder/temp/encoded/encoded_compress_{size_conversion}-{file_extension}.npy"
recon_img_path = f"M:/codes/codes_BAM/python/autoencoder/temp/decoded/decoded_compress_{size_conversion}-{file_extension}.npy"

if file_extension[-3:] == "CNN":
    correction = 0
else:
    correction = 0.5 
    
orig_img = np.load(orig_img_path); orig_img = orig_img[:,:,:,0] + correction
encoded_images = np.load(compressed_img_path) + correction
decoded_images = np.load(recon_img_path); decoded_images = decoded_images[:,:,:,0] + correction

#%% Visualize images and compare
num_images_to_show = 1
for im_ind in range(num_images_to_show):
    plot_ind = im_ind*3 + 1
    #rand_ind = np.random.randint(low=0, high=orig_img.shape[0])
    rand_ind = int(orig_img.shape[0]/2)
    
    plt.subplot(num_images_to_show, 3, plot_ind)
    plt.imshow(orig_img[rand_ind, :, :], cmap="gray")
    plt.title("Original")
      
    plt.subplot(num_images_to_show, 3, plot_ind+1)
    plt.imshow(encoded_images[rand_ind, :, :, 0], cmap="gray")
    plt.title("One encoded layer")
    
    plt.subplot(num_images_to_show, 3, plot_ind+2)
    plt.imshow(decoded_images[rand_ind, :, :], cmap="gray")
    plt.title("Decoded")
    
    plt.subplots_adjust(wspace=0.4, hspace=5)
    #plt.savefig(f"M:/codes/codes_BAM/python/autoencoder/temp/figures/8bit_comparison/visual_comparison_{size_conversion}-{file_extension}", dpi=600)
#%% Calculate metrics
total_vol = orig_img.size

MSE = np.sum(np.absolute(orig_img - decoded_images))/orig_img.size
PSNR = (20*np.log10(255))-(10*np.log10(MSE))

#%% Threshold images
orig_img_int8 = (orig_img*255).astype("uint8")
decoded_images_int8 = (decoded_images*255).astype("uint8")

t_val_orig = np.zeros(orig_img.shape[0])
orig_img_thresh  = np.zeros((orig_img.shape))
t_val_decoded = np.zeros(orig_img.shape[0])
decoded_img_thresh  = np.zeros((orig_img.shape))

for i in range(orig_img.shape[0]):
    t_val_orig[i], orig_img_thresh[i,:,:] = cv.threshold(orig_img_int8[i,:,:], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    t_val_decoded[i], decoded_img_thresh[i,:,:] = cv.threshold(decoded_images_int8[i,:,:], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

#%% Visualize thresholded images and compare
plt.subplot(num_images_to_show, 2, 1)
plt.imshow(orig_img_thresh[rand_ind, :, :], cmap="gray")
plt.title("Original thresholded")
    
plt.subplot(num_images_to_show, 2, 2)
plt.imshow(decoded_img_thresh[rand_ind, :, :], cmap="gray")
plt.title("Decoded thresholded")
    
plt.subplots_adjust(wspace=0.4, hspace=5)
plt.savefig(f"M:/codes/codes_BAM/python/autoencoder/temp/figures/8bit_comparison/visual_comparison_thresholded_{size_conversion}-{file_extension}", dpi=600)

#%%
MSE_thresh = np.sum(np.absolute(orig_img_thresh - decoded_img_thresh))/orig_img.size
PSNR_thresh = -10*np.log10(MSE)