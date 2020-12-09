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
import features

#-----------------------------------------------------------------------------

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
    (number of images specified by num_images)                                                      
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
    new_data = torch.zeros((t[0], new_dim_len, t[1], t[2], t[3]))
    
    for i in range(N):
        
        imgs = [data[i]]
        
        for j in range(len(filters)):
            
            filtered_img = features.filter_img(data[i], filters[j])
            imgs.append(filtered_img)
        
        if (contrast):
            
            contrasted_img = features.contrast_img(data[i], factor)
            imgs.append(contrasted_img)
            
        imgs_tensor = torch.stack(imgs)
        new_data[i] = imgs_tensor
        
    return new_data
            
    
#-----------------------------------------------------------------------------

def train(model, criterion, train_set, train_gts, test_set, test_gts,
          optimizer, scheduler, num_epochs, batch_size=16):
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
            
            # clear the gradients
            optimizer.zero_grad()

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
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

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy(prediction, batch_y))
            
        
        train_accuracy = sum(accuracies_train).item()/len(accuracies_train) 
        test_accuracy = sum(accuracies_test).item()/len(accuracies_test)
        print("Epoch {} | Train accuracy: {:.5f} and test accuracy: {:.5f}" \
              .format(epoch+1, train_accuracy, test_accuracy))
        
        
    print ("Training completed")

#-----------------------------------------------------------------------------

class NeuralNet(torch.nn.Module):
    
    def __init__(self):
      
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 32, kernel_size=(1,3,3))
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=(1,3,3))
        self.conv2_drop = torch.nn.Dropout3d(p=0.5)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        #print(x.shape)
        relu = torch.nn.functional.relu
        max_pool3d = torch.nn.functional.max_pool3d
        x = relu(max_pool3d(self.conv1(x), (1,2,2)))
        #print(x.shape)
        x = relu(max_pool3d(self.conv2_drop(self.conv2(x)), (3,2,2)))
        #print(x.shape)
        x = x.view(-1, 256)
        x = relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    

#-----------------------------------------------------------------------------
    
#%%
size_train_set = 50

data, labels = load_data(size_train_set, w=16, h=16, seed=1)

print(data.shape)

new_data = add_features(data, ['unsharp'], contrast=True, factor=2.5)

print(new_data.shape)

train_imgs, train_gts, test_imgs, test_gts = split_data(new_data, labels, 0.7)


"""
c0 = 0  # bgrd
c1 = 0  # road
for i in range(len(train_gts)):
    if train_gts[i] == 0:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

print('Balancing training data...')
min_c = min(c0, c1)
idx0 = [i for i, j in enumerate(train_gts) if j == 0]
idx1 = [i for i, j in enumerate(train_gts) if j == 1]
new_indices = idx0[0:min_c] + idx1[0:min_c]
print("Training on {a} batches instead of {b}".format(a=len(new_indices),
                                                     b=train_imgs.shape[0]))
train_imgs = train_imgs[new_indices, :, :, :]
train_gts = train_gts[new_indices]

train_size = train_gts.shape[0]

c0 = 0
c1 = 0
for i in range(len(train_gts)):
    if train_gts[i] == 0:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
"""



t = train_imgs.shape
s = test_imgs.shape

train_imgs = train_imgs.reshape((t[0], t[4], t[1], t[2], t[3]))
test_imgs = test_imgs.reshape((s[0], s[4], s[1], s[2], s[3]))

print(train_imgs.shape)



#%%
model = NeuralNet()

num_epochs = 30
learning_rate = 0.001

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
train(model, criterion, train_imgs, train_gts, test_imgs, test_gts,
      optimizer, scheduler, num_epochs, 16)

#%%

train(model, criterion, train_imgs, train_gts, test_imgs, test_gts,
      optimizer, scheduler, num_epochs)







