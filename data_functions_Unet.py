#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:44:55 2020

@author: cyrilvallez
"""

import torch
import numpy as np

import images

#-----------------------------------------------------------------------------

def load_data(num_images, seed=1):
    """
    Return tensors of images and correspondind groundtruths in the
    correct shape
    img_tor : shape = (num_images, 3, 400, 400)
    gts_tor : shape = (num_images, 400, 400)
    """
    imgs, gts = images.load_nimages(num_images, seed=seed)
    
    img_torch = torch.stack(imgs)
    gts_torch = torch.stack(gts)
    
    gts_torch = gts_torch.round().long()
    img_torch = img_torch.permute(0, 3, 1, 2)
    
    
    return img_torch, gts_torch

#-----------------------------------------------------------------------------

def get_prediction(output):
    """
    Get prediction image from the output of the U-net
    output : shape = (1, 2, 400, 400)
    prediction : shape = (400, 400)
    """
    t = output.shape
    output = output.view(t[1], t[2], t[3])
    
    prediction = torch.argmax(output, 0)
    
    return prediction

#-----------------------------------------------------------------------------

def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    
    @param predicted_logits: float32 tensor of shape (1, 2, 400, 400)
    @param reference: int64 tensor of shape (400,400) with the labels
    """
    prediction = get_prediction(predicted_logits)
    correct_predictions = prediction.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()

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

#-----------------------------------------------------------------------------

def train(model, criterion, train_set, train_gts, test_set, test_gts,
          optimizer, scheduler, device, num_epochs, batch_size=1):
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
    
    model = model.to(device)
    
    N_batch = int(train_set.shape[0]/batch_size)
    N_batch_test = int(test_set.shape[0]/batch_size)
    
    train_size = train_set.shape[0]
    training_indices = range(train_size)
    test_size = test_set.shape[0]
    testing_indices = range(test_size)
     
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        accuracies_train = []
        accuracies_test = []
        
        # Permute training indices
        perm_indices = np.random.permutation(training_indices)
        
        for i in range(N_batch):
            
            batch_indices = perm_indices[i*batch_size:(i+1)*batch_size]
            
            batch_x = train_set[batch_indices]
            batch_y = train_gts[batch_indices]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # clear the gradients
            optimizer.zero_grad()

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
        
            # Compute the gradient
            loss.backward()
        
            # Update the parameters of the model with a gradient step
            optimizer.step()
            
            with torch.no_grad():
                accuracies_train.append(accuracy(prediction, batch_y))
            
        # Make a scheduler step
        #scheduler.step()
            
        # Test the quality on the test set
        model.eval()
        
        perm_indices_test = np.random.permutation(testing_indices)
        
        for i in range(N_batch_test):
            
            batch_indices_test = perm_indices_test[i*batch_size:(i+1)*batch_size]
            
            batch_x = test_set[batch_indices_test]
            batch_y = test_gts[batch_indices_test]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            with torch.no_grad():

                # Evaluate the network (forward pass)
                prediction = model(batch_x)
                accuracies_test.append(accuracy(prediction, batch_y))
            
        
        train_accuracy = sum(accuracies_train).item()/len(accuracies_train) 
        test_accuracy = sum(accuracies_test).item()/len(accuracies_test)
        print("Epoch {} | Train accuracy: {:.5f} and test accuracy: {:.5f}" \
              .format(epoch+1, train_accuracy, test_accuracy))
        
        
    print ("Training completed")

#-----------------------------------------------------------------------------

