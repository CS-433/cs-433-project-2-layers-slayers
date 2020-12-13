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
num_subgroups = 50
range_subgroup = imgs.shape[0]//num_subgroups
for i in range(num_subgroups-1):
  preds[range_subgroup*i:range_subgroup*(i+1)] = ML.validation.get_prediction(model(imgs[range_subgroup*i:range_subgroup*(i+1)]), False)
print(preds.shape)

test_dir = "data/test_set_images/"
files = np.array(sorted(os.listdir(test_dir)))
for i in range(len(files)):
  Imaging.save_img(preds[i],files[i],'output/')
  

submission_filename = 'first_try.csv'
image_filenames = []
for i in range(1,51):
  image_filename = 'output/test_' + '%d' % i + '.png'
  image_filenames.append(image_filename)

helpers.masks_to_submission(submission_filename, *image_filenames)
  
  
  
  
  
  
  
  
  
  
  
  
  
  