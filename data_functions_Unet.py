#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:44:55 2020
@author: cyrilvallez
"""

import torch
import numpy as np

import images
import validation
import features

def load_data_BCE_loss(num_images, rotate=True, angles=[90, 180, 270], seed=1):
    """
    Return tensors of images and correspondind groundtruths in the
    correct shape
    img_tor : shape = (num_images, 3, 400, 400)
    gts_tor : shape = (num_images, 400, 400)
    """
    imgs, gts = images.load_nimages(num_images, seed=seed)
    
    img_torch = torch.stack(imgs)
    gts_torch = torch.stack(gts)
    
    gts_torch = gts_torch.round()
    img_torch = img_torch.permute(0, 3, 1, 2)
    
    t = gts_torch.shape
    gts_new = torch.zeros(t[0], t[1], t[2], 2)
    
    gts_new[gts_torch == 0] = torch.tensor([1., 0.])
    gts_new[gts_torch ==1] = torch.tensor([0.,1.])
    gts_new = gts_new.permute(0,3,1,2)
    
    if rotate:
        
        rotated_imgs = img_torch
        rotated_gts = gts_new
        
        print ("Starting rotations")
        
        for i in range(len(angles)):
            rot_imgs = features.rotate(img_torch, angles[i])
            rot_gts = features.rotate(gts_new, angles[i])
            
            rotated_imgs = torch.cat((rotated_imgs, rot_imgs), 0)
            rotated_gts = torch.cat((rotated_gts, rot_gts), 0)
            
        print ("Done !")
        
        img_torch = rotated_imgs
        gts_new = rotated_gts 
            
    return img_torch, gts_new

#-----------------------------------------------------------------------------

def get_prediction(output, mask):
    """
    Get prediction image from the output of the U-net for a batch of size 1
    output : shape = (1, 2, 400, 400)
    prediction : shape = (1, 400, 400)
    """
    
    prediction = torch.argmax(output, 1)
    imgheight = output.shape[1]
    imgwidth = output.shape[2]
    
    threshold = 0.25
    w = 16
    h = 16
    
    if (mask):
        
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                    patch = prediction[:, j:j+w, i:i+h].float()
                    if (patch.mean() >= threshold):
                        prediction[:, j:j+w, i:i+h] = 1
                    else:
                        prediction[:, j:j+w, i:i+h] = 0
                    
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

def train(model, criterion, train_set, train_gts,
          optimizer, scheduler, device, num_epochs, batch_size=1,testing=False,
          test_set=torch.empty(0), test_gts=torch.empty(0)):
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
    
    N_batch = int(train_set.shape[0]/batch_size)
    if testing:
        N_batch_test = int(test_set.shape[0]/batch_size)
    
    train_size = train_set.shape[0]
    training_indices = range(train_size)
    
    if testing:
        test_size = test_set.shape[0]
        testing_indices = range(test_size)
     
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        accuracies_train = []
        accuracies_test = []
        f1_test = []
        
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
                pred = get_prediction(prediction, True)
                gt = get_prediction(batch_y, False)  ##### REMOVE AFTER TEST
                acc, _ = validation.accuracy(pred, gt)  ### HERE TOO
                accuracies_train.append(acc)
            
        # Make a scheduler step
        #scheduler.step()
        
        if testing:
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
                    pred = get_prediction(prediction, True)
                    gt = get_prediction(batch_y, False)  ### REMOVE AFTER TEST
                    acc, _ = validation.accuracy(pred, gt)  ## HERE TOO
                    accuracies_test.append(acc)
                    f1_test.append(validation.f1_score(pred, gt))  ## HERE TOO
                
            
            train_accuracy = sum(accuracies_train).item()/len(accuracies_train) 
            test_accuracy = sum(accuracies_test).item()/len(accuracies_test)
            test_f1 = sum(f1_test).item()/len(f1_test)
            print("Epoch {} | Train accuracy: {:.5f} || test accuracy: {:.5f} || test f1: {:.5f}" \
                  .format(epoch+1, train_accuracy, test_accuracy, test_f1))
        else:
            train_accuracy = sum(accuracies_train).item()/len(accuracies_train)
            print("Epoch {} | Train accuracy: {:.5f}".format(epoch+1,train_accuracy))
        
        
    print ("Training completed")
    return test_accuracy

#-----------------------------------------------------------------------------