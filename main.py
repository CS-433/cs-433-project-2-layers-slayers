# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:35:40 2020

@author: Darius
__________
Main file to run learning process.
"""

import torch

import Imaging
import MachineLearning as ML
############################################################################
# MAIN    

size_train_set = 10
rotation_angles = [0,45,315]
data, labels = Imaging.load_data(size_train_set, rotate=True, angles=rotation_angles)

# crop_data = torch.empty(0)
# for ind in range(data.shape[0]):
#     crop_img = torch.stack(Imaging.crop_img(data[ind]))
#     crop_data = torch.cat((crop_data, crop_img), 0)
    
# crop_labels = torch.empty(0).long()
# for ind in range(data.shape[0]):
#     crop_img = torch.stack(Imaging.crop_img(labels[ind]))
#     crop_labels = torch.cat((crop_labels, crop_img), 0)


ratio = 0.7
train_imgs, train_gts, test_imgs, test_gts = Imaging.split_data(data, labels, ratio)



# If a GPU is available (should be on Colab, we will use it)
# if not torch.cuda.is_available():
#  raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 25
learning_rate = 0.001

model_name = f'UNet_{size_train_set}_{num_epochs}'
filters = ['edge','edge+']

model = ML.model.UNet()
# model = ML.model.LogisticRegression()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

tr_acc, te_acc, te_f1 = ML.training.train(model, criterion, train_imgs, train_gts,
                                          optimizer, scheduler, device, num_epochs, batch_size=1,
                                          testing=True, test_set=test_imgs, test_gts=test_gts)

torch.save(model.state_dict(), f'saved-models/{model_name}.pt')


















