# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:30:41 2025
---
Evaluate compressed and reconstructed image quality as compared to original 8 bit images

Comparing the following quantities:
1. Calculated porosity percentages
2. Mean asuare differences of Sobel filtered images

Additionally, have side by side comparison and the difference image
---
@author: bhejazi
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#%% Load output images from compression of binary images
size_conversion = "512x512-to-128x128x8"
file_extension = "binary-images-CNN"

orig_img_path = f"yourfilepath_{file_extension}.npy"
compressed_img_path = f"yourfilepath_{size_conversion}-{file_extension}.npy"
recon_img_path = f"yourfilepath_{size_conversion}-{file_extension}.npy"

if file_extension[-3:] == "CNN":
    correction = 0
else:
    correction = 0.5 
    
orig_img = np.load(orig_img_path); orig_img = orig_img[:,:,:,0] + correction
encoded_images = np.load(compressed_img_path) + correction
decoded_images = np.load(recon_img_path); decoded_images = decoded_images[:,:,:,0] + correction

#%% Visualize images
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
    #plt.savefig(f"yourfilepath_{size_conversion}-{file_extension}", dpi=600)

#%% Threshold images
decoded_images_int8 = (decoded_images*255).astype("uint8")

t_val_decoded = np.zeros(orig_img.shape[0])
decoded_img_thresh  = np.zeros((orig_img.shape))

for i in range(orig_img.shape[0]):
    t_val_decoded[i], decoded_img_thresh[i,:,:] = cv.threshold(decoded_images[i,:,:], 0.5, 1, cv.THRESH_BINARY)

#%% Visualize thresholded images
plt.imshow(decoded_img_thresh[rand_ind, :, :], cmap="gray")
plt.title("Decoded thresholded")

#%% Calculate porosity
total_vol = orig_img.size

orig_img_pore_pct = 100*np.count_nonzero(orig_img)/total_vol
decoded_pore_pct = 100*np.count_nonzero(decoded_img_thresh)/total_vol

#%% Calculate PSNR

MSE = np.sum(np.absolute(orig_img - decoded_img_thresh))/orig_img.size

PSNR = -10*np.log10(MSE)

#%% Apply sobel (lagranigian) filter and compare
orig_img_lap  = np.zeros((orig_img.shape))
decoded_img_lap = np.zeros((orig_img.shape))

for i in range(orig_img.shape[0]):
    orig_img_lap[i,:,:] = cv.Laplacian(orig_img[i,:,:], cv.CV_64F)
    orig_img_lap[i,:,:] = np.absolute(orig_img_lap[i,:,:])
    orig_img_lap[i,:,:] = np.uint8(orig_img_lap[i,:,:])
    
    t_val, orig_img_lap[i,:,:] = cv.threshold(orig_img_lap[i,:,:], 0, 1, cv.THRESH_BINARY)
    
    decoded_img_lap[i,:,:] = cv.Laplacian(decoded_img_thresh[i,:,:], cv.CV_64F)
    decoded_img_lap[i,:,:] = np.absolute(decoded_img_lap[i,:,:])
    decoded_img_lap[i,:,:] = np.uint8(decoded_img_lap[i,:,:])
    
    t_val, decoded_img_lap[i,:,:] = cv.threshold(decoded_img_lap[i,:,:], 0, 1, cv.THRESH_BINARY)

edge_diffs = np.absolute(orig_img_lap-decoded_img_lap)
edge_diff_pct = 100*np.count_nonzero(edge_diffs)/total_vol

# Plot the Laplacian edge detection for both original and decoded images
plt.subplot(num_images_to_show, 3, 1)
plt.imshow(orig_img_lap[rand_ind, :, :], cmap="gray")
plt.title("Original Laplacian")
    
plt.subplot(num_images_to_show, 3, 2)
plt.imshow(decoded_img_lap[rand_ind, :, :], cmap="gray")
plt.title("Decoded Laplacian")
    
plt.subplot(num_images_to_show, 3, 3)
plt.imshow(edge_diffs[rand_ind, :, :], cmap="gray")
plt.title("Diff image")

plt.subplots_adjust(wspace=0.5, hspace=5)
plt.savefig(f"yourfilepath_{size_conversion}-{file_extension}", dpi=600)


