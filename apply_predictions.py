#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:12:30 2020

@author: alangmeier
"""

import torch
import MachineLearning as ML
import Imaging

model = ML.model.UNet3D()
model.load_state_dict(torch.load('saved-models/UNet.pt', map_location=torch.device('cpu')))
model.eval()

imgs = Imaging.load_test()
print(imgs.shape)

preds = model(imgs)
print(preds.shape)