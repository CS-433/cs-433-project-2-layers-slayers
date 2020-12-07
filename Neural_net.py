#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:26:03 2020

@author: cyrilvallez
"""

import torch
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

import images


def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0

#------------------------------------------------------------------------------

def load_data(num_images, w=16, h=16, seed=1):
    """
    Creates tensor of the mini-batches of all the images 
    (number of images specified by num_images)                                                      
    """
    imgs, gts = images.load_nimages(num_images, seed=seed)

    num_images = len(imgs)

    img_patches = [images.crop_img(imgs[i], w, h) for i in range(num_images)]
    data_img = np.asarray([img_patches[i][j] for i in range(len(img_patches)) \
                       for j in range(len(img_patches[i]))])
        
    gts_patches = [images.crop_img(gts[i], w, h) for i in range(num_images)]
    data_gt = np.asarray([gts_patches[i][j] for i in range(len(gts_patches)) \
                       for j in range(len(gts_patches[i]))])
    labels = np.asarray(
        [value_to_class(np.mean(data_gt[i])) for i in range(len(data_gt))])

    return torch.from_numpy(data_img), torch.from_numpy(labels)

#------------------------------------------------------------------------------

def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    
    @param predicted_logits: float32 tensor of shape (batch size, 2)
    @param reference: int64 tensor of shape (batch_size) with the labels
    """
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


#-----------------------------------------------------------------------------

def train(model, criterion, set_train, set_gt, optimizer, num_epochs,
          batch_size=16):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.loss._Loss
    @param set_train: tensor of mini-batches from different images
                      shape = (mini_batch_size, 16, 16, 3)
    @param set_gt: tensor of mini-batches from different gt images
                      shape = (mini_batch_size, 16, 16)
    @param optimizer: torch.optim.Optimizer
    @param num_epochs: int
    """
    
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        N_batch = int(set_train.shape[0]/batch_size)
        accuracies_train = []
        
        for i in range(N_batch):
            
            batch_x = set_train[i*batch_size:(i+1)*batch_size]
            batch_y = set_gt[i*batch_size:(i+1)*batch_size]

            # Evaluate the network (forward pass)
            prediction = model.forward(batch_x)
            loss = criterion(prediction, batch_y)
            accuracies_train.append(accuracy(prediction, batch_y))
        
            # Compute the gradient
            optimizer.zero_grad()  
            loss.backward()
        
            # Update the parameters of the model with a gradient step
            optimizer.step()
            
        print("Epoch {} | Training accuracy: {:.5f}".format(epoch, 
                        sum(accuracies_train).item()/len(accuracies_train)))
        
        
    print ("Training completed")
    



#-----------------------------------------------------------------------------

class NeuralNet(torch.nn.Module):
    
    def __init__(self):
      
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d(p=0.5)
        self.fc1 = torch.nn.Linear(20, 10)
        self.fc2 = torch.nn.Linear(10, 2)

    def forward(self, x):
        relu = torch.nn.functional.relu
        max_pool2d = torch.nn.functional.max_pool2d
        x = relu(max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)
        x = x.view(16, 20)
        x = relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    

#-----------------------------------------------------------------------------

size_train_set = 20

imgs, gts = load_data(size_train_set, w=16, h=16)

t = imgs.shape

imgs = imgs.reshape((t[0], t[3], t[1], t[2]))

model = NeuralNet()

num_epochs = 10
learning_rate = 1e-3

criterion = torch.nn.CrossEntropyLoss() # this includes LogSoftmax which executes a logistic transformation

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train(model, criterion, imgs, gts, optimizer, num_epochs)



