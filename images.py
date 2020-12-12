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
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import features


# ____________________________ Loading images ____________________________
def load_data(num_images, rotate=False, flip=False, angles=[0,90,180,270], directions=[0,1,2,3],seed=1):
    """
    Return tensors of images and correspondind groundtruths in the
    correct shape
    Augment the data set with rotation and flip if wanted
    img_tor : shape = (num_images, 3, 400, 400)
    gts_tor : shape = (num_images, 400, 400)
    """
    imgs, gts = load_nimages(num_images, seed=seed)
    
    img_torch = torch.stack(imgs)
    gts_torch = torch.stack(gts)
    
    gts_torch = gts_torch.round().long()
    img_torch = img_torch.permute(0, 3, 1, 2)
    
    if (rotate or flip):
        if not(rotate):
            angles = [0]
        if not(flip):
            directions = [0]
        
        
        augmented_imgs = torch.empty(0)
        augmented_gts = torch.empty(0)
        
        print ("Starting rotations and/or flip")
        
        for angle in angles:
            for direction in directions:
                trans_imgs = img_torch
                trans_gts = gts_torch
                if rotate:
                    trans_imgs = features.rotate(trans_imgs, angle)
                    trans_gts = features.rotate(trans_gts, angle)
                if flip:
                    trans_imgs = features.flip(trans_imgs, direction)
                    trans_gts = features.flip(trans_gts, direction)
                
                    
                augmented_imgs = torch.cat((augmented_imgs, trans_imgs), 0)
                augmented_gts = torch.cat((augmented_gts, trans_gts), 0)
            
        print ("Done !")
        
        img_torch = augmented_imgs
        gts_torch = augmented_gts 
            
    return img_torch, gts_torch



def load_image(infilename:str):
    """ Loads the image specified by the filename (string) passed as argument.
        __________
        Parameters : filename (str)
        Returns : image (torch tensor) 
    """
    
    data = mpimg.imread(infilename)
    return torch.from_numpy(data)

# def load_image(i:int):
#     """ Loads image i (int) contained in the training dataset (from 1 to 100).
#         Parameters : i (int)
#         Returns : img (array of floats) """
    
#     image_dir = "data/training/images/"
#     image_filename = "satImage_%.3d.png" % i
    
#     data = mpimg.imread(image_dir + image_filename)
#     return data

def load_nimages(n, train=True, filters=None, seed=1):
    """ Loads n (int) images in an arbitrary order and returns a list of these 
        and another of their corresponding groundtruths if the argument train 
        is set to True (default). Otherwise, it returns one list of test images.
        It is also possible to precise a list of filters with values specified 
        inside method features.filter_img (default: None).
        __________
        Parameters : n (int), train=True (boolean), filters=None (list of str)
        Returns : images (list of torch tensors), 
                  groundtruths (list of torch tensors) (if train=True),
                  filtered images (list of torch tensors) (if filters!=None)
    """
        
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
        Parameters : image (torch tensor or Image object), image_name (str), path='./' (str)
        Returns : None 
    """
    
    if type(img) == torch.Tensor:
        plt.imsave(path + image_name + ".png", img.numpy())
        
    elif type(img) == Image.Image:
        img.save(path + image_name + ".png")

# ____________________________ Extracting patches ____________________________

def crop_img(im, w=16, h=16):
    """ Extracts patches of the image passed as argument with the specified 
        width and length (default : 16x16).
        __________
        Parameters : image (torch tensor), width=16 (int), height=16 (int)
        Returns : patches of image (list of torch tensors) 
    """
    
    list_patches = []
    imgwidth = im.size()[0]
    imgheight = im.size()[1]
    is_2d = len(im.size()) < 3
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
        Parameters : image (torch tensor), groundtruth (torch tensor) 
        Returns : concatenated image (Image object) 
    """
    
    nChannels = len(gt_img.size())
    w = gt_img.size()[0]
    h = gt_img.size()[1]
    if nChannels == 3:
        cimg = torch.cat((img, gt_img), 1)
    else:
        gt_img_3c = torch.zeros((w, h, 3), dtype=torch.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = torch.cat((img8, gt_img_3c), 1)
    return Image.fromarray(cimg.numpy())


def overlay_images(img, predicted_img):
    """ Overlays an image and its groundtruth so that the groundtruth image is 
        a red slightly transparent mask.
        __________
        Parameters : image (torch tensor), groundtruth (torch_tensor)
        Returns : overlayed image (Image object)
    """
    
    img = img.permute(1,2,0)
    w = img.size()[0]
    h = img.size()[1]
    color_mask = torch.zeros((w, h, 3), dtype=torch.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8.numpy(), 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask.numpy(), 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# ____________________________ Conversion functions ____________________________

def img_float_to_uint8(img):
    """ Converts the image passed as argument from float to uint8.
        __________
        Parameters : image (torch tensor of float)
        Returns : image (torch tensor of uint8) """
    
    rimg = img - torch.min(img)
    rimg = (rimg / torch.max(rimg) * 255).round().type(torch.ByteTensor)
    
    return rimg

def convert_to_tensor(img):
    """ Converts the image passed as argument from PIL Image to Pytorch tensor.
        __________
        Parameters : image (PIL Image)
        Returns : image (torch tensor)
    """
    
    if type(img) == Image.Image:
        converter = transforms.ToTensor()
        return converter(img).permute(1,2,0)
    elif type(img) == torch.Tensor:
        return img
    else:
        raise TypeError("Cannot convert this type to Pytorch tensor.")
        
def convert_to_Image(img):
    """ Converts the image passed as argument from Pytorch tensor to PIL Image.
        __________
        Parameters : image (torch tensor)
        Returns : image (PIL Image)
    """
    
    if type(img) == torch.Tensor:
        converter = transforms.ToPILImage()
        return converter(img.permute(2,0,1))
    elif type(img) == Image.Image:
        return img
    else:
        raise TypeError("Cannot convert this type to PIL Image.")