# Road Segmentation
Machine Learning (CS-433) : Project 2.

## Authors
- Darius Nik Nejad (<darius.niknejad@epfl.ch>)
- Cyril Vallez (<cyril.vallez@epfl.ch>)
- André Langmeier (<andre.langmeier@epfl.ch>)

## Folders
- data : *contains all the data used to train and test the models*.
- saved-models : *contains all the models we used for our kept AIcrowd prediction*
- Imaging, MachineLearning, helpers : *contain the internal packages used for the project : helpers, Imaging and MachineLearning*

## Libraries and Packages
External Libraries
- NumPy
- Matplotlib
- PyTorch (needs to be installed)
- PIL
- os,sys
- SciPy
- pickle
- random

Internal Libraries
- helpers : *contains functions to help with the submission format of AIcrowd*
- Imaging : *contains all the functions related to image data processing (more details below)*
- MachineLearning : *contains all the functions and classes related to the implemented models (more details below)*

## Code organization
Imaging
- The Imaging library contains functions to load the images in tensor format and some functions for the data processing (cropping, splitting in train/validation sets, etc.). It also contains functions that analyses predictions visually.
- The features sub-module contains the data enhancement functions for rotations and flipping, and for the filters.

MachineLearning
- The MachineLearning library is divided into three sub-modules : model, training and validation
- The model sub-module contains all the model classes (torch.nn.module) used in the project
- The validation sub-module contains functions to measure the efficiency of a model, as well as functions returning predictions
- The training sub-module contains the training functions

Running the code
- To reproduce the setup we used for our AIcrowd prediction, simply run the script *run.py*
