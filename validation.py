#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 2020

@author: alangmeier
__________
This file contains methods that allow to evaluate if a model is a good one or 
not, by implementing accuracies computations and other methods of validation.
"""

# ____________________________ IMPORTS ____________________________
import numpy as np
import torch

# ____________________________ Accuracies computation ____________________________

def accuracy(predictions, groundtruths):
    """ Computes the overall accuracy, i.e. the number of true positives. It also
        computes the detailed accuracy, i.e. the true-/false-/-positives/-negatives.
        __________
        Parameters : predictions (list of torch tensors), corresponding 
                     groundtruths (list of torch tensors)
        Returns : overall accuracy (float), detailed acccuracy (dictionary of float values)
                  the dictionary contains entries : 'tp' (true positive), 'fp' (false positive)
                                                    'fn' (false negative), 'tn' (true negative)
    """
    
    if type(predictions) == torch.Tensor:
        predictions = [predictions]
    if type(groundtruths) == torch.Tensor:
        groundtruths = [groundtruths]
    
    N = 0
    detailed_accuracy = {"tp" : 0.0,
                         "fp" : 0.0,
                         "fn" : 0.0,
                         "tn" : 0.0}
    
    for i in range(len(predictions)):
        pred = predictions[i].float().round()
        gt = groundtruths[i].float().round()
        assert pred.size() == gt.size()
        
        tp_i = torch.sum(torch.logical_and(pred==1, gt==1))
        fp_i = torch.sum(torch.logical_and(pred==1, gt==0))
        fn_i = torch.sum(torch.logical_and(pred==0, gt==1))
        tn_i = torch.sum(torch.logical_and(pred==0, gt==0))
        
        N += torch.numel(pred)
        detailed_accuracy["tp"] += tp_i
        detailed_accuracy["fp"] += fp_i
        detailed_accuracy["fn"] += fn_i
        detailed_accuracy["tn"] += tn_i
        
    detailed_accuracy["tp"] = detailed_accuracy["tp"] / N
    detailed_accuracy["fp"] = detailed_accuracy["fp"] / N
    detailed_accuracy["fn"] = detailed_accuracy["fn"] / N
    detailed_accuracy["tn"] = detailed_accuracy["tn"] / N
    overall_accuracy = detailed_accuracy["tp"] + detailed_accuracy["tn"]
        
    
    return overall_accuracy, detailed_accuracy
    

# ____________________________ F1-Score computation ____________________________

def f1_score(predictions, groundtruths):
    """ Computes the F1-Score obtained for the predictions and their corresponding 
        groundtruths. The computation of this score is given by :
        F1 = (true positive) / (true positives + 0.5*(false positives + false negatives))
        __________
        Parameters : predictions (list of torch tensors), corresponding 
                     groundtruths (list of torch tensors)
        Returns : F1-Score (float)
    """
    
    _ , acc = accuracy(predictions, groundtruths)
    
    return (acc["tp"]) / (acc["tp"] + 0.5*(acc["fp"] + acc["fn"]))
     
    
    
    