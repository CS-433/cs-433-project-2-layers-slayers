#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:26:03 2020

@author: cyrilvallez
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import images

def train(model, criterion, set_train, set_gt, optimizer, num_epochs):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.loss._Loss
    @param set_train: list of tensors of shape (400,400,3)
    @param set_gt: list of tensors of shape (400,400)
    @param optimizer: torch.optim.Optimizer
    @param num_epochs: int
    """
    
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        
        for i in range(len(set_train)):
            
            split_train = images.crop_img(set_train[i])
            split_gt = images.crop_img(set_gt[i])
            
            
            for j in range(len(split_train)):
        
                batch_x = split_train[j]
                batch_y = split_gt[j]

                # Evaluate the network (forward pass)
                prediction = model.forward(batch_x)
                loss = criterion(prediction, batch_y)
        
                # Compute the gradient
                optimizer.zero_grad()  
                loss.backward()
        
                # Update the parameters of the model with a gradient step
                optimizer.step()
        
    print ("Training completed")
    



##############################################################################

class NeuralNet(torch.nn.Module):
    
    def __init__(self):
      
        super().__init__()
        self.conv1 =  torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d(p=0.5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 2)

    def forward(self, x):
        relu = torch.nn.functional.relu
        max_pool2d = torch.nn.functional.max_pool2d

        x = relu(max_pool2d(self.conv1(x), 2))
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    

##############################################################################
    
    
    

train_set, train_labels = images.load_nimages(2)

train_set = [torch.from_numpy(train_set[i]) for i in range(len(train_set))]
train_labels = [torch.from_numpy(train_labels[i]) for i in range(len(train_labels))]


img_patches = [images.crop_img(train_set[i], 16, 16) for i in range(2)]
data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]


