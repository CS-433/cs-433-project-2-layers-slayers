#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 2020

@author: alangmeier
__________
This files contains methods to compute/improve the feature inputs used in the
models. Also contains methods to rotate and flip images.
"""

# ____________________________ IMPORTS ____________________________
import torch
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as functional
import scipy.ndimage as sc

import Imaging as images

# ____________________________ Filtering ____________________________
# Filters used come from the PIL.ImageFilter library so it must handle PIL Image.
# Therefore, the only allowed input types are PIL Image or Pytorch tensors. 
# The returned type is a Pytorch tensor, which is more appropriate for networks.

def filter_img(img,filt):
    """ Applies the specified filter to the image.
        __________
        Parameters : image (torch tensor or PIL Image), filter (str)
        Returns : image (torch tensor)
    """
    
    if type(img) == torch.Tensor:
        img = images.convert_to_Image(img)
    if type(img) != Image.Image:
        raise TypeError("Cannot apply filters to this type.")
    
    filters = {
        'edge+' : ImageFilter.FIND_EDGES,
        'edge' : ImageFilter.EDGE_ENHANCE_MORE,
        'contour' : ImageFilter.CONTOUR,
        'emboss' : ImageFilter.EMBOSS,
        'blur' : ImageFilter.BLUR,
        'unsharp' : ImageFilter.UnsharpMask
        }
    
    rimg = img.filter(filters[filt])
    rimg = images.convert_to_tensor(rimg)
    return rimg


# ____________________________ Enhancing ____________________________

def contrast_img(img,factor=2.5):
    """ Applies an enhancement of the image by adjusting its contrast by the 
        specified factor (default: 2.5). A factor of 1.0 returns the same image.
        __________
        Parameters : image (torch tensor or PIL Image), factor=2.5 (float)
        Returns : image (torch tensor)
    """
    
    if type(img) == torch.Tensor:
        img = images.convert_to_Image(img)
    if type(img) != Image.Image:
        raise TypeError("Cannot apply filters to this type.")
        
    enhancer = ImageEnhance.Contrast(img)
    rimg = enhancer.enhance(factor)
    rimg = images.convert_to_tensor(rimg)
    return rimg
    
    
# ____________________________ Adding features ____________________________
def add_features (data, filters, contrast=True, factor=2.5):
    """
    Creates one more dimension on the data containing the images filtered
    data : 4D tensor of the patches images
    filters : list of strings containing the wanted filters
    contrast : boolean (True for contrast filter, False otherwise)
    factor : contrast factor (1 return original image)
    """
    
    N = data.shape[0]
    new_dim_len = len(filters) + 1   # +1 for original image
    if (contrast):
        new_dim_len += 1
        
    t = data.shape
    new_data = torch.zeros((t[0], new_dim_len, t[1], t[2], t[3]), device=data.device)
    
    for i in range(N):
        
        imgs = [data[i]]
        
        for j in range(len(filters)):
            
            filtered_img = filter_img(data[i], filters[j]).to(data.device)
            imgs.append(filtered_img)
        
        if (contrast):
            
            contrasted_img = contrast_img(data[i], factor).to(data.device)
            imgs.append(contrasted_img)
            
        imgs_tensor = torch.stack(imgs)
        new_data[i] = imgs_tensor
        
    return new_data
    

def cat_features (data, filters, contrast=True, factor=2.5):
    """ Adds the filtered images along the channels' dimension.
        __________
        Parameters : data (4D tensor of images with channels at dim=3), filters (list of str),
                     contrast (boolean), factor (float)
        Returns : augmented data (4D tensor with increased channel dimension)
    """
    
    t = data.shape
    N = t[0]
    new_channels_dim = 3*(1+len(filters))
    if contrast:
        new_channels_dim += 3
    new_data = torch.zeros(t[0], t[1], t[2], new_channels_dim, device=data.device)
    
    for i in range(N):
        imgs = data[i]
        for j in range(len(filters)):
            imgs = torch.cat((imgs,filter_img(data[i], filters[j]).to(data.device)),dim=2)
        
        if contrast:
            imgs = torch.cat((imgs,contrast_img(data[i],factor).to(data.device)),dim=2)            
        
        new_data[i] = imgs
        
    return new_data
        
    
# ____________________________ Augmenting set ____________________________

def rotate(img, angle):
    """
    rotate a single tensor image or tensor of tensor images by an angle angle
    img : shape = (N, C, W, H)
    return a single tensor image or tensor of tensor images of rotated images
    return shape = (N, C, W, H)
    """
    Ndim = len(img.shape)
    rot = sc.rotate(img, angle, (Ndim-2,Ndim-1), reshape=False, order=0, \
                    mode='mirror')
    return torch.from_numpy(rot)

def flip(img,direction):
    """
    fip a single tensor image or tensor of tensor images horizontally
    (direction=1) or vertically (direction=2) or both (direction=3)
    img : shape = (N, C, W, H)
    return a single tensor image or tensor of tensor images of flipped images
    return shape = (N, C, W, H)
    """
    if direction == 0:
        return img
    elif direction == 1:
        return functional.hflip(img)
    elif direction == 2:
        return functional.vflip(img)
    else:
        return functional.vflip(functional.hflip(img))
    
    
# ____________________________ Standardizing data ____________________________

def standardize(img):
    """ Standardizes each channel of the input image, i.e. mean=0 and std=1.
        __________
        Parameters : images (torch Tensor) of dim (N,C,H,W)
        Returns : images (torch Tensor) with the same dimensions and each channel standardized.
    """
    
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    
    rimg = torch.zeros_like(img)
    for i in range(img.shape[0]):
        for channel in range(3):
            rimg[i,channel] = (img[i,channel] - torch.mean(img[i,channel])) / torch.std(img[i,channel])
            
    
    return rimg.squeeze(0)
    
    
    
    
    
    
    
    
    
    
    
    
