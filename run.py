# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:30:57 2020

@author: Darius
__________
This script should output a submission close to what we got.
"""

import time
import torch

import Imaging
import MachineLearning as ML
##############################################################################
# Training the algorithm




####################################################### UNet2D
size_train_set = 100
rotation_angles = [0,15,30,45,60,75]
directions = [0,1,2]
data, labels = Imaging.load_data(size_train_set, rotate=True, flip=True, angles=rotation_angles, directions=directions)

# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
    raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 4
learning_rate = 0.001

model = ML.model.UNet().to(device)
model_name = 'UNet'

c1 = labels.sum().item()
c0 = labels.nelement() - c1
total = labels.nelement()
w0 = 1. - c0/total
w1 = 1. - c1/total
weights = torch.tensor([w0, w1]).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.90)

scheduler = 0 #we finally did not use it

ML.Training.k_cross_train(k, model, criterion, data, labels, optimizer, scheduler,
                  [50,50,50,50], device, batch_size=1, split_indicies=[],
                  save_model = True, model_name = model_name ,epoch_freq_save = 1)



################################### Filtered UNet2D
size_train_set = 100
rotation_angles = [0,15,45,315]
directions = [0,3]
data, labels = Imaging.load_data(size_train_set, rotate=True, flip=True, angles=rotation_angles, directions=directions)

# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
    raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 4
learning_rate = 0.001

model = ML.model.UNet().to(device)
model_name = 'UNet2D_filtered'

weights = torch.tensor([0.25,0.75]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

scheduler = 0 #we finally did not use it

ML.Training.k_cross_train(k, model, criterion, data, labels, optimizer, scheduler,
                  [40,15,15,15], device, batch_size=1, split_indicies=[],
                  save_model = True, model_name = model_name ,epoch_freq_save = 5)


################################### UNet3D
size_train_set = 100
rotation_angles = [0,15,30,45,195, 315]
data, labels = Imaging.load_data(size_train_set, rotate=True, flip=False, angles=rotation_angles)

# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
    raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 4
learning_rate = 0.001

model = ML.model.UNet().to(device)
model_name = 'UNet3D'

weights = torch.tensor([0.25,0.75]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

scheduler = 0 #we finally did not use it

ML.Training.k_cross_train(k, model, criterion, data, labels, optimizer, scheduler,
                  [40,15,15,15], device, batch_size=1, split_indicies=[],
                  save_model = True, model_name = model_name ,epoch_freq_save = 5)