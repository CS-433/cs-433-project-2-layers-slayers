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
import itertools
import matplotlib.pyplot as plt

import Imaging

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
    

# ____________________________ F1-Score computation __________________________

def f1_score(tp,fp,fn):
    """ Computes the F1-Score :
        F1 = (true positive) / (true positives + 0.5*(false positives + false negatives))
        __________
        Parameters : tp (#true positive), fp (#false positive), fn (#false negative)
        Returns : F1-Score (float)
    """
    
    
    return tp / (tp + 0.5*(fp + fn))

# ______________________ Prediction with several models ______________________

def sum_prediction_models(model_type, models_filenames_list, batch_x, printage = False):
    """ Returns the sum of prediction given a list of models (there is a road if the
        sum of each prediction per model is above sensitivity)
        __________
        Parameters :
            model_type : instance of the class of the models
            models_list : list of models' filenames
            batch_x : set of images, shape : (N,C,W,H)   
    """
    prediction = 0
    
    for model_name in models_filenames_list:
        if printage:
            print(model_name)
        model_type.load_state_dict(torch.load(f'saved-models/{model_name}.pt'))
        with torch.no_grad():
            prediction += get_prediction(model_type(batch_x), False)

    return prediction




def predict_with_models(models_list, list_model_names, list_filters, list_type_UNet, batch_x, device, sensitivity = 1):
    """ Returns a prediction given a list of models (there is a road if the
        sum of each prediction per model is above sensitivity)
        __________
        Parameters :
            model_type : instance of the class of the models
            models_list : list of models' filenames
            batch_x : set of images, shape : (N,C,W,H)     
            sensitivity : sensitivity of prediction
    """
    prediction_sum_batch = 0  
    for i in range(len(list_model_names)):
        if len(list_filters[i]) > 0:
            if list_type_UNet[i] == '3D':
                batch_x = Imaging.features.add_features(batch_x, list_filters[i], False)
            else:
                batch_x = Imaging.features.cat_features (batch_x, list_filters[i], False)
        batch_x = batch_x.to(device)
                
        model = models_list[i]
        with torch.no_grad():
            prediction_sum_batch += sum_prediction_models(model, list_model_names[i], batch_x, False)

    return (prediction_sum_batch >= sensitivity).long()



def sub_lists(my_list):
    """ 
    Returns a list of all sublists of liste
    """
    subs =[]
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in itertools.combinations(my_list, i)]
        if len(temp)>0:
            subs.extend(temp)
    return subs




def best_combination_of_models(models_list, list_model_names, list_filters, list_type_UNet, data, labels, device, validation_type = 'F1', printage = True, print_frequency = 50):
    """ Returns the combination of models yielding the highest validation on
        the training set (see predict_with_models)
        __________
        Parameters :
            models_list : list of instances of models
            list_predictions_list : list of list of model names (one list per model)
            data : validation set
            labels : groundtruths of validation set
            validation_type : a validation method (F1 or accuracy)
            printage : print or not modulo 100
    """
    validation_best = 0
    sensitivity_best = 0
    combination_best = []
    mod_combinations = sub_lists(list(range(len(list_model_names))))
    del mod_combinations[0]
    
    
    f1_scores = []
    accuracies = []
    for combination in mod_combinations: #loop over all the possible combinations of models
        print(combination)        
        
        prediction_sum = torch.empty(0, dtype=torch.int64).to(device)
        for j in range(data.shape[0]): #mini batch the data, not to explose RAM
            if printage:
                if (j+1)%print_frequency == 0:
                    print(j+1)
            prediction_sum_batch = 0  
            for i in combination:
                batch_x = data[j:j+1]
                if len(list_filters[i]) > 0:
                    if list_type_UNet[i] == '3D':
                        batch_x = Imaging.features.add_features(batch_x, list_filters[i], False)
                    else:
                        batch_x = Imaging.features.cat_features (batch_x, list_filters[i], False)
                batch_x = batch_x.to(device)
                
                model = models_list[i]
                with torch.no_grad():
                    prediction_sum_batch += sum_prediction_models(model, list_model_names[i], batch_x, False)
            
            prediction_sum = torch.cat((prediction_sum,prediction_sum_batch),0)
        
        names = [list_model_names[i] for i in combination]
        concatenated = [j for i in names for j in i]
        for sensitivity in range(1,len(concatenated)+1): #loop over all the possible sensitivities       
            prediction = (prediction_sum >= sensitivity).long().to('cpu')
            acc, details = accuracy(prediction, labels)
            tp = details['tp']
            fp = details['fp']
            fn = details['fn']
            f1 = f1_score(tp, fp, fn)
            
            accuracies.append(acc)
            f1_scores.append(f1)
            print("Models {} with sensitivity {} || Accuracy: {} | F1 Score : {}".format(combination,sensitivity,acc,f1))
            
            if validation_type == 'accuracy':
                validation_meas = acc
            else:
                validation_meas = f1
            
            if validation_meas > validation_best: #to track the model with higher validation
                validation_best = validation_meas
                combination_best = combination
                sensitivity_best = sensitivity
    
    return combination_best, sensitivity_best, validation_best, accuracies, f1_scores

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

# ___________________ PLot the tracks (accuracy and f1) ______________________
    
def plot_kfold_track(list_of_tracklists, list_of_colours = [], linetype = '.-'):
    """
    Plot the tracks for a k-fold training
    __________
    Parameters :
        list_of_tracklists : list of list of tracks (each list = 1 fold)
        list_of_colours : list of colours for each fold
    """
    n_after = 0
    for i in range(len(list_of_tracklists)):
        n_before = n_after + 1
        n_after += len(list_of_tracklists[i])
        epochs = list(range(n_before,n_after+1))
        
        if len(list_of_colours)==len(list_of_tracklists):
            plt.plot(epochs,list_of_tracklists[i],list_of_colours[i]+linetype)
        else:
            plt.plot(epochs,list_of_tracklists[i],linetype)
        
        