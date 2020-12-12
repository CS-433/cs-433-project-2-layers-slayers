# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:05:52 2020

@author: Darius
__________
This file contains Machine Learning models (as classes) as well as the main train 
function and k-fold cross validation.
"""


#TODO : comment code and functions
import torch
import numpy as np
import random

import features

##############################################################################
# MODELS
    
class NeuralNet(torch.nn.Module): # ~79%
    
    def __init__(self):
      
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = torch.nn.Dropout2d(p=0.5)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        relu = torch.nn.functional.relu
        max_pool2d = torch.nn.functional.max_pool2d
        x = relu(max_pool2d(self.conv1(x), 2)) 
        #print(x.shape)
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2)) 
        #print(x.shape)
        x = x.reshape(-1, 256)
        x = relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    
#_____________________________________________________________________________
class LogisticRegression(torch.nn.Module): # ~74%
    def __init__(self):
        super().__init__()
        #EVENTUALLY PUT PIXELS AS PARAMETERS
        self.num_pixels = 16
        self.image_volume = 3 * self.num_pixels * self.num_pixels #colours x pixels 
        self.num_classes = 2 #num_classes (one answer per image)
        self.linear_transform = torch.nn.Linear(self.image_volume, self.num_classes, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        flattened_images = x.reshape(batch_size, self.image_volume)
        return self.linear_transform(flattened_images)
    
    
#_____________________________________________________________________________
class Conv3DNet(torch.nn.Module): # ~83%
    def __init__(self):
        super().__init__()
        #EVENTUALLY PUT PIXELS AS PARAMETERS
        self.filters = ['edge','edge+']
        self.num_pixels = 16
        self.depth = 1+len(self.filters)
        self.image_volume = 3 * self.num_pixels * self.num_pixels * self.depth #colours x pixels 
        self.num_classes = 2 #num_classes (one answer per image)
        
        self.conv1 = torch.nn.Conv3d(3, 32, kernel_size=3, padding=(1,1,1))
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=3, padding=(1,1,1))
        # self.conv3 = torch.nn.Conv3d(64, 32, kernel_size=3, padding=(1,1,1))
        self.lin1 = torch.nn.Linear(self.depth*1024, self.depth*128)
        self.lin2 = torch.nn.Linear(self.depth*128, self.num_classes)
        
        
    def forward(self,x):
        relu = torch.nn.functional.relu
        max_pool3d = torch.nn.functional.max_pool3d
        device = x.device
        
        filtered_x = []
        for filt in self.filters:
            filtered_x.append(torch.stack([features.filter_img(x.permute(0,2,3,1)[i],filt).to(device) 
                                                   for i in range(x.shape[0])
                                                           ]).permute(0,3,1,2))
        
        x = torch.stack([x,*filtered_x],dim=2)
        
        x = relu(max_pool3d(self.conv1(x),(1,2,2), padding=0))
        x = relu(max_pool3d(self.conv2(x),(1,2,2), padding=0))
        # x = relu(max_pool3d(self.conv3(x),(1,2,2),padding=0))
        x = x.view(-1, self.depth*1024)
        x = relu(self.lin1(x))
        x = relu(self.lin2(x))
        return x
    
    
#_____________________________________________________________________________
    

##############################################################################
# k Cross validation

#useful functions

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def separate_train_test(list_ind,i,dataset):
    list_rest = list_ind.copy()
    del list_rest[i]
    list_rest = [item for sublist in list_rest for item in sublist]
    dataset_test = [dataset[0][list_ind[i]],dataset[1][list_ind[i]]]
    dataset_train = [dataset[0][list_rest],dataset[1][list_rest]]
    return dataset_train, dataset_test


#k fold cross validation function for 1 model
#TODO : keep the model at the end + verify if model is reset each time
def k_cross_validation(k, model, criterion, images_set, groundtruths, optimizer, scheduler, num_epochs, device, batch_size=16):
    dataset = [images_set, groundtruths]
    list_ind = partition(list(range(dataset[0].shape[0])),k)
    accuracy = 0.
    for i in range(k):
        print("Fold: {}/{}".format(i+1,k))
        dataset_train, dataset_test = separate_train_test(list_ind,i,dataset)
        accuracy += 1/k*train(model, criterion, dataset_train[0], dataset_train[1], dataset_test[0], dataset_test[1], optimizer, scheduler, num_epochs, device, batch_size)
    return accuracy


##############################################################################
# TRAINING FUNCTION

def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    
    @param predicted_logits: float32 tensor of shape (batch size, 2)
    @param reference: int64 tensor of shape (batch_size) with the labels
    """
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()



def train(model, criterion, train_set, train_gts, test_set, test_gts,
          optimizer, scheduler, num_epochs, device, batch_size=16):
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
            with torch.no_grad():
                accuracies_train.append(accuracy(prediction, batch_y))
        
            # Compute the gradient
            loss.backward()
        
            # Update the parameters of the model with a gradient step
            optimizer.step()
            
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

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy(prediction, batch_y))
            
        
        train_accuracy = sum(accuracies_train).item()/len(accuracies_train) 
        test_accuracy = sum(accuracies_test).item()/len(accuracies_test)
        print("Epoch {} | Train accuracy: {:.5f} and test accuracy: {:.5f}" \
              .format(epoch+1, train_accuracy, test_accuracy))
        
        
    print ("Training completed")
    return test_accuracy
    

    
    
    
    
    
    
    