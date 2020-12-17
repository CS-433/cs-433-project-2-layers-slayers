#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:12:30 2020

@author: alangmeier
"""

import numpy as np
import os 
import torch
import MachineLearning as ML
import Imaging
import helpers


model = ML.model.UNet3D().cuda()
model.load_state_dict(torch.load('saved-models/UNet.pt'))
model.eval()

imgs = Imaging.load_test().cuda()
print(imgs.shape)

preds = torch.zeros(imgs.shape[0],imgs.shape[2],imgs.shape[3])

with torch.no_grad():
    for i in range(imgs.shape[0]):
      out = model(imgs[i:i+1])
      pred = ML.validation.get_prediction(out, True)
      pred = pred.view(pred.shape[1], pred.shape[2])
      preds[i] = pred

## Optionnaly save the predictions -------------------------------------------
test_dir = "data/test_set_images/"
N = 50
files = ['test_' + '%d' %i for i in range(1,N+1)]
for i in range(len(files)):
  Imaging.save_img(preds[i],files[i],'output/')
##----------------------------------------------------------------------------


submission_filename = 'first_try.csv'

helpers.masks_to_submission(submission_filename, preds.numpy())
  
  
  
  
  
  
  
  
  
  
  
  
  
  