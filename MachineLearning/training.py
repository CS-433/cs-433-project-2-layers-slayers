# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:37:32 2020

@author: Darius
__________
This file contains the training function. Also contains training with k-fold
cross validation.
"""

import torch
import numpy as np
import random
import pickle

from MachineLearning import validation

# ----------------------------------------------------------------------------
def train(model, criterion, train_set, train_gts,
          optimizer, scheduler, device, num_epochs, batch_size=1,testing=False,
          test_set=torch.empty(0), test_gts=torch.empty(0)):
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
    if testing:
        train_accuracy_epoch = []
        test_accuracy_epoch = []
        test_f1_epoch = []
        N_batch_test = int(test_set.shape[0]/batch_size)
    
    train_size = train_set.shape[0]
    training_indices = range(train_size)
    
    if testing:
        test_size = test_set.shape[0]
        testing_indices = range(test_size)
        
    model = model.to(device)
     
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        accuracies_train = []
        
        # Permute training indices
        perm_indices = np.random.permutation(training_indices)
        
        for i in range(N_batch):
            
            batch_indices = perm_indices[i*batch_size:(i+1)*batch_size]
            
            batch_x = train_set[batch_indices]
            batch_y = train_gts[batch_indices]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # clear the gradients
            optimizer.zero_grad()

            # Evaluate the network (forward pass)
            prediction = model(batch_x)          
            #print(prediction.dtype,batch_y.dtype)
            loss = criterion(prediction, batch_y)
        
            # Compute the gradient
            loss.backward()
        
            # Update the parameters of the model with a gradient step
            optimizer.step()
            
            
            with torch.no_grad():
                pred = validation.get_prediction(prediction, False)
                gt = validation.get_prediction(batch_y, False)  ##### REMOVE AFTER TEST
                acc, _ = validation.accuracy(pred, gt)  ### HERE TOO
                accuracies_train.append(acc)
            
        # Make a scheduler step
        #scheduler.step()
        
        if testing:
            accuracies_test = []
            
            # Test the quality on the test set
            model.eval()
            
            perm_indices_test = np.random.permutation(testing_indices)
            tp = 0;
            fp = 0;
            fn = 0;
            for i in range(N_batch_test):
                
                batch_indices_test = perm_indices_test[i*batch_size:(i+1)*batch_size]
                
                batch_x = test_set[batch_indices_test]
                batch_y = test_gts[batch_indices_test]
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                with torch.no_grad():
    
                    # Evaluate the network (forward pass)
                    prediction = model(batch_x)
                    pred = validation.get_prediction(prediction, False)
                    gt = validation.get_prediction(batch_y, False)  ### REMOVE AFTER TEST
                    acc, details = validation.accuracy(pred, gt)  ## HERE TOO
                    accuracies_test.append(acc)
                    tp += details['tp']
                    fp += details['fp']
                    fn += details['fn']
                    
            f1_test = validation.f1_score(tp,fp,fn).item()   
            train_accuracy = sum(accuracies_train).item()/len(accuracies_train) 
            test_accuracy = sum(accuracies_test).item()/len(accuracies_test)
            print("Epoch {} | Train accuracy: {:.5f} || test accuracy: {:.5f} || test f1: {:.5f}" \
                  .format(epoch+1, train_accuracy, test_accuracy, f1_test))
                
            train_accuracy_epoch.append(train_accuracy)
            test_accuracy_epoch.append(test_accuracy)
            test_f1_epoch.append(f1_test)
        else:
            train_accuracy = sum(accuracies_train).item()/len(accuracies_train)
            print("Epoch {} | Train accuracy: {:.5f}".format(epoch+1,train_accuracy))
        
        
    print ("Training completed")
    if testing:
        return train_accuracy_epoch, test_accuracy_epoch, test_f1_epoch
    
# ---------------------------------------------------------------------------

def partition (list_in, n):
    """
    Randomly paritition list_in in n categories
    """
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def separate_train_test(list_ind,i,dataset):
    """
    Separate dataset in one dataset_test (where indicies are list_ind[i]) and
    one training_dataset (the remaining indices).
    Each dataset is a list containing 2 tensors : [images, groundtruths]
    """
    list_rest = list_ind.copy()
    del list_rest[i]
    list_rest = [item for sublist in list_rest for item in sublist]
    dataset_test = [dataset[0][list_ind[i]],dataset[1][list_ind[i]]]
    dataset_train = [dataset[0][list_rest],dataset[1][list_rest]]
    return dataset_train, dataset_test

def k_cross_train(k, model, criterion, dataset, gts, optimizer, scheduler,
                  num_epochs_list, device, batch_size=1, keep_models=True):
    """
    Train the model with k-fold cross validation. Returns a track of the
    accurcies and F1 scores, for each training.
    
    num_epochs_list is a list containing num_epochs for fold (make first
    element greater). If keep_models is True, also returns a list of the
    copied models after each training.
    """  
      
    dataset = [dataset, gts]
    list_ind = partition(list(range(dataset[0].shape[0])),k)
    
    train_accuracy_epoch_k = []
    test_accuracy_epoch_k = []
    test_f1_epoch_k = []
    models_list = []
    for i in range(k):
        print("Fold: {}/{}".format(i+1,k))
        dataset_train, dataset_test = separate_train_test(list_ind,i,dataset)
        
        train_accuracy_epoch, test_accuracy_epoch, test_f1_epoch = train(
            model, criterion, dataset_train[0], dataset_train[1], optimizer,
            scheduler, device, num_epochs_list[i], batch_size, True,
            dataset_test[0], dataset_test[1])
        
        train_accuracy_epoch_k.append(train_accuracy_epoch)
        test_accuracy_epoch_k.append(test_accuracy_epoch)
        test_f1_epoch_k.append(test_f1_epoch)
        
        if keep_models:
            model_copy =  pickle.loads(pickle.dumps(model))
            models_list.append(model_copy)
        
    if keep_models:
        return models_list, train_accuracy_epoch_k, test_accuracy_epoch_k, test_f1_epoch_k
    else:
        return train_accuracy_epoch_k, test_accuracy_epoch_k, test_f1_epoch_k