# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:58:12 2020

@author: Darius
"""

import torch
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

import MLmodel
import images
import Class_UNet as unet
import data_functions_Unet as unet_utils
import features
import UNet3D


#############################################################################
#useful functions

def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = v.sum()
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0

#------------------------------------------------------------------------------

def load_data(num_images, w=16, h=16, seed=1):
    """
    Creates tensor of the mini-batches of all the images 
    (number of images specified by num_images).                                              
    """
    
    imgs, gts = images.load_nimages(num_images, seed=seed)

    num_images = len(imgs)

    img_patches = [images.crop_img(imgs[i], w, h) for i in range(num_images)]
    imgs_list = [img_patches[i][j] for i in range(len(img_patches)) \
                       for j in range(len(img_patches[i]))]
        
    gts_patches = [images.crop_img(gts[i], w, h) for i in range(num_images)] 
    gts_list = [gts_patches[i][j] for i in range(len(gts_patches)) \
                        for j in range(len(gts_patches[i]))] 
    labels = [value_to_class(gts_list[i].mean()) for i in range(len(gts_list))]
    

    return torch.stack(imgs_list), torch.tensor(labels)

#-----------------------------------------------------------------------------

def split_data(data, labels, ratio, seed=1):
    """split the dataset based on the split ratio."""
    
    np.random.seed(seed)
    
    N = data.shape[0]
    shuffle = np.random.permutation(np.arange(N))
    shuffled_data = data[shuffle]
    shuffled_labels = labels[shuffle]
    
    stop = round(ratio*N)
    
    data_training = shuffled_data[0:stop]
    labels_training = shuffled_labels[0:stop]
    data_testing = shuffled_data[stop:N]
    labels_testing = shuffled_labels[stop:N]

    return data_training, labels_training, data_testing, labels_testing



############################################################################
# MAIN    

size_train_set = 10

# data, labels = load_data(size_train_set)
data, labels = unet_utils.load_data(size_train_set)

div = int(data.shape[0]/2)

ratio = 0.7
# train_imgs, train_gts, test_imgs, test_gts = split_data(data, labels, ratio)
train_imgs, train_gts, test_imgs, test_gts = unet_utils.split_data(data, labels, ratio)

train_imgs = train_imgs.permute(0,2,3,1)
train_list = [features.filter_img(train_imgs[i], 'edge') for i in range(train_imgs.shape[0])]
train_list = torch.stack(train_list)
train_imgs = train_list.permute(0,3,1,2)

test_imgs = test_imgs.permute(0,2,3,1)
test_list = [features.filter_img(test_imgs[i], 'edge') for i in range(test_imgs.shape[0])]
test_list = torch.stack(test_list)
test_imgs = test_list.permute(0,3,1,2)

# train_imgs = train_imgs.permute(0,3,1,2)
# test_imgs = test_imgs.permute(0,3,1,2)


# If a GPU is available (should be on Colab, we will use it)
# if not torch.cuda.is_available():
#  raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



num_epochs = 15
learning_rate = 0.001
k = 4

model_name = 'UNet3D'
filters = ['edge','contour']

model = UNet3D.UNet3D(filters)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# MLmodel.train(model, criterion, train_imgs, train_gts, test_imgs, test_gts, optimizer, scheduler, num_epochs, device)
unet_utils.train(model, criterion, train_imgs, train_gts, test_imgs, test_gts, optimizer, scheduler, device, num_epochs)

# fold4_accuracy = MLmodel.k_cross_validation(k, model, criterion, data, labels, optimizer, scheduler, num_epochs, device)

torch.save(model.state_dict(), f'saved-models/{model_name}.pt')















