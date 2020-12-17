# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:30:57 2020

@author: Darius
__________
This script should output a submission close to what we got.
"""

import time
import torch

import helpers
import Imaging
import MachineLearning as ML


#___________________________TRAINING__________________________________________

# We did not obtain our submissions this way. We run the code on several days
# on several computers.This is why this training part is a bit long.
# We also came back to previous folds manually to improve them (in k-fold cross
# training). This code should however gives somewhat close to what we got.

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
name = 'UNet'

c1 = labels.sum().item()
c0 = labels.nelement() - c1
total = labels.nelement()
w0 = 1. - c0/total
w1 = 1. - c1/total
weights = torch.tensor([w0, w1]).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.90)

scheduler = 0 #we finally did not use it

ML.training.k_cross_train(k, model, criterion, data, labels, optimizer, scheduler,
                  [50,50,50,50], device, batch_size=1, split_indicies=[],
                  save_model = True, name = name ,epoch_freq_save = 5)

del data, labels
del model, optimizer, criterion

################################### Filtered UNet2D
size_train_set = 100
rotation_angles = [0,15,45,315]
directions = [0,3]
data, labels = Imaging.load_data(size_train_set, rotate=True, flip=True, angles=rotation_angles, directions=directions)
data = Imaging.features.cat_features (data, ['unsharp'], False)

# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
    raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 4
learning_rate = 0.001

model = ML.model.UNet(1).to(device)
name = 'UNet2D_filtered'


c1 = labels.sum().item()
c0 = labels.nelement() - c1
total = labels.nelement()
w0 = 1. - c0/total
w1 = 1. - c1/total
weights = torch.tensor([w0, w1]).to(device)


criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

scheduler = 0 #we finally did not use it

ML.training.k_cross_train(k, model, criterion, data, labels, optimizer, scheduler,
                  [60,25,25,25], device, batch_size=1, split_indicies=[],
                  save_model = True, name = name ,epoch_freq_save = 5)


del data, labels
del model, optimizer, criterion




#___________________________SUBMISSION________________________________________

# As explained in the report we use the two types of network, and each of their
# k models to provide the best submission. The evaluation of the best combination
# is not made here, but the function that does it is in ML.validation !

imgs = Imaging.load_test()

list_model_names = [[f'UNet_k0',
                    f'UNet_k1',
                    f'UNet_k2',
                    f'UNet_k3'],
                    [f'UNet2D_filtered_k0',
                    f'UNet2D_filtered_k1',
                    f'UNet2D_filtered_k2',
                    f'UNet2D_filtered_k3']]

models_list = [ML.model.UNet().to(device), ML.model.UNet(1).to(device)]

models_list[0].eval()
models_list[1].eval()

list_filters = [[],['unsharp']]
list_type_UNet = ['2D','2D']

sensitivity = 4

preds = torch.zeros(imgs.shape[0], 608, 608)
for i in range(preds.shape[0]):
    if (i+1)%5 == 0:
      print(i+1)
    batch_x = imgs[i:i+1]
    pred = ML.validation.predict_with_models(models_list, list_model_names, list_filters, list_type_UNet, batch_x, device, sensitivity)
    pred = pred.view(pred.shape[1], pred.shape[2])
    preds[i] = pred

submission_filename = 'FinalSubmission.csv'

helpers.masks_to_submission(submission_filename, preds.numpy())