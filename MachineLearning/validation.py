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
import torch
import pickle

# ____________________________ Get prediction ________________________________
def get_prediction(output, mask, w=16, h=16, threshold=0.25):
    """
    Get prediction image from the output of the U-net for a batch of size 1
    output : shape = (1, 2, 400, 400)
    prediction : shape = (1, 400, 400)
    """
    if len(output.shape)==3:
        prediction = output
    else:
        prediction = torch.argmax(output, 1)
    
    if (mask):
        imgheight = prediction.shape[1]
        imgwidth = prediction.shape[2] 
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                    patch = prediction[:, j:j+w, i:i+h].float()
                    if (patch.mean() >= threshold):
                        prediction[:, j:j+w, i:i+h] = 1
                    else:
                        prediction[:, j:j+w, i:i+h] = 0
                    
    return prediction

# ____________________________ Accuracies computation ________________________

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

def f1_score(tp,fp,fn):
    """ Computes the F1-Score :
        F1 = (true positive) / (true positives + 0.5*(false positives + false negatives))
        __________
        Parameters : tp (#true positive), fp (#false positive), fn (#false negative)
        Returns : F1-Score (float)
    """
    
    
    return tp / (tp + 0.5*(fp + fn))

# ______________________ Prediction with several models ______________________

def predict_with_models(model_type, models_filenames_list, batch_x, sensitivity = 1):
    """ Returns a prediction given a list of models (there is a road if the
        sum of each prediction per model is above sensitivity)
        __________
        Parameters :
            model_type : instance of the class of the models
            models_list : list of models' filenames
            batch_x : set of images, shape : (N,C,W,H)     
            sensitivity : sensitivity of prediction
    """
    prediction = 0
    
    for model_name in models_filenames_list:
        torch.save(model_type.state_dict(), f'saved-models/{model_name}.pt')
        with torch.no_grad():
            prediction += get_prediction(model_type(batch_x), False)
        
    return (prediction >= sensitivity).long()


def predict_with_predictions(predictions_list, sensitivity = 1):
    """ Returns a prediction given a list of predictions (there is a road if the
        sum of each prediction is above sensitivity)
        __________
        Parameters :
            predictions_list : list of predictions  
            sensitivity : sensitivity of prediction
    """
    return (sum(predictions_list) >= sensitivity).long()


# ______________________ Save and load a list ______________________

def save_list(list_to_save,file_name):
    """
    Saves the list to file_name
    """
    open_file = open(file_name, "wb")
    pickle.dump(list_to_save, open_file)
    open_file.close()

def load_list(file_name):
    """
    Loads and returns the list from file_name
    """
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list
