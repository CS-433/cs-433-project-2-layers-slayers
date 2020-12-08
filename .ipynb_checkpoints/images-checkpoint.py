#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 5 2020

@author: alangmeier
__________
This file contains methods to load images, converting them, cropping them as 
well as arranging them for plotting.
"""

# ____________________________ IMPORTS ____________________________
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image


# ____________________________ Loading images ____________________________

def load_image(infilename:str):
    """ Loads the image specified by the filename (string) passed as argument.
        __________
        Parameters : filename (str)
        Returns : image (array) """
    
    data = mpimg.imread(infilename)
    return data

# def load_image(i:int):
#     """ Loads image i (int) contained in the training dataset (from 1 to 100).
#         Parameters : i (int)
#         Returns : img (array of floats) """
    
#     image_dir = "data/training/images/"
#     image_filename = "satImage_%.3d.png" % i
    
#     data = mpimg.imread(image_dir + image_filename)
#     return data

def load_nimages(n, train=True, seed=1):
    """ Loads n (int) images in an arbitrary order and returns a list of these 
        and another of their corresponding groundtruths if the argument train 
        is set to True (default). Otherwise, it returns one list of test images.
        __________
        Parameters : n (int), train=True (boolean)
        Returns : imgs (list), groundtruths (list) (if train=True) """
        
    if train:
        root_dir = "data/training/"
        image_dir = root_dir + "images/"
        gt_dir = root_dir + "groundtruth/"
    
        files = np.array(sorted(os.listdir(image_dir)))
        np.random.seed(seed) # Seeding so everyone gets the same samples
        random_indices = np.random.permutation(len(files))
        files = files[random_indices]
        n = min(n, len(files)) # Load maximum n images
        
        print("Loading " + str(n) + " images")
        imgs = [load_image(image_dir + files[i]) for i in range(n)]
        
        print("Loading " + str(n) + " corresponding groundtruths")
        gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    
        return imgs, gt_imgs
    
    else:
        root_dir = "data/test_set_images/"
        
        files = np.array(sorted(os.listdir(root_dir)))
        np.random.seed(1) # Seeding so everyone gets the same samples
        random_indices = np.random.permutation(len(files))
        files = files[random_indices]
        n = min(n, len(files))
        
        print("Loading " + str(n) + " images")
        imgs = [load_image(root_dir + files[i] + "/" + f"{files[i]}.png") 
                for i in range(n)]
        
        return imgs


# ____________________________ Saving images ____________________________

def save_img(img, image_name, path='./'):
    """ Saves the image passed as argument with the specified image_name at the 
        specified path (default: current folder) with 'png' format.
        __________
        Parameters : image (array), image_name (str), path='.' (str)
        Returns : None """
        
    plt.imsave(path + image_name + ".png", img)

# ____________________________ Extracting patches ____________________________

def crop_img(im, w=16, h=16):
    """ Extracts patches of the image passed as argument with the specified 
        width and length (default : 16x16).
        __________
        Parameters : image (array), width=16 (int), height=16 (int)
        Returns : patches of image (list) """
    
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# ____________________________ Arranging images for plotting ____________________________

def concatenate_images(img, gt_img):
    """ Concatenates an image and its groundtruth along the horizontal axis.
        __________
        Parameters : image (array), groundtruth (array) 
        Returns : concatenated image (array) """
    
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def overlay_images(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# ____________________________ Helper functions ____________________________

def img_float_to_uint8(img):
    """ Converts the image passed as argument from float to uint8.
        __________
        Parameters : image (array of float)
        Returns : image (array of uint8) """
    
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


