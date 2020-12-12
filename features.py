#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 2020

@author: alangmeier
__________
This files contains methods to compute/improve the feature inputs used in the
models.
"""

# ____________________________ IMPORTS ____________________________
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as functional
from PIL import Image, ImageFilter, ImageEnhance

import images

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


#-----------------------------------------------------------------------------

def rotate(img, angle):
    """
    rotate a single tensor image or tensor of tensor images by an angle angle
    img : shape = (N, C, W, H)
    return a single tensor image or tensor of tensor images of rotated images
    return shape = (N, C, W, H)
    """
    return functional.rotate(img, angle)

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