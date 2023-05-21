# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img, # Maybe remove
                                                  img_to_array, # Maybe remove
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input, # Maybe remove
                                                 decode_predictions, # ;aybe remove
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, # Maybe remove
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer # Maybe remove
from sklearn.metrics import classification_report

# Data Munging
import numpy as np
import shutil
import pandas as pd
# For plotting
import matplotlib.pyplot as plt
# System tools
import os
import zipfile
import argparse
import sys
sys.path.append("utils")

#Helper functions
from helper_functions import image_generators, pretrained_model, model_training, predicting, classifier_report_save, plots, model_save



## argparse 
def input_parse():
    # Command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, help = "Path to the zip folder")
    parser.add_argument("--epochs", type = int, default = 10, help = "How many epochs, default is 10")
    args = parser.parse_args()

    return args


def unzip(args):
    folder_path = os.path.join("data", "Test_Alphabet") # Path to the data if unzipped already
    if not os.path.exists(folder_path): # Checking to see if folder is unzipped
        print("Unzipping file")
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = os.path.join("data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination)
    print("The files are unzipped")


## Deleting folders
def delete():
    print("Deleting folders...")
    folder_path_train = os.path.join("data","Train_Alphabet") # Folder path train
    folder_path_test = os.path.join("data","Test_Alphabet") # Folder path test
    folders_to_delete = ["Blank"]  # List of folders to delete

    for folder_name in folders_to_delete: # For every index in the list above 
        folder_path = os.path.join(folder_path_train, folder_name)
        if os.path.exists(folder_path): # If the path exists
            shutil.rmtree(folder_path) # Delete the folder and its contents 
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"Folder: {folder_path} is already deleted")
        
        folder_path = os.path.join(folder_path_test, folder_name) # The same for test folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"Folder: {folder_path} is already deleted")  
    return folder_path_train, folder_path_test
    

def data_generator(datagen, folder_path_train, train_or_val):
    print("Creating train and validation images")
    generator = datagen.flow_from_directory( # Using flow from directory and the train ImageDataGenerator
        folder_path_train, # Folderpath to train images
        target_size=(90, 90), # Load the images as 90 by 90 pixels
        color_mode='rgb', # Colour images
        class_mode='categorical', # One hot encode labels
        batch_size=32, # Batch size 32
        shuffle=True, # Shuffle the images 
        subset=train_or_val, # Subset (specificed in mainfunction)
        interpolation='nearest') # Rescaling function
    return generator


def test_data_generator(test_datagen, folder_path_test):
    test_tensorflow = test_datagen.flow_from_directory(
        folder_path_test, # Folder path to test images
        target_size=(90, 90), # Load the images as 90 by 90 pixels
        color_mode='rgb', # Colour images 
        batch_size=32, # batch size of 32
        shuffle=False, # Do not shuffle
        interpolation='nearest') # Rescaling function
    return test_tensorflow


def classifier_report(test_tensorflow, training_tensorflow, pred):
    print("Creating classification report")
    true_labels = test_tensorflow.classes # Getting the labels 
    # Mapping the labels from numerical values to label names 
    label_map = {v: k for k, v in training_tensorflow.class_indices.items()}
    # Converting the labels 
    true_labels = np.array([label_map[label] for label in true_labels])
    # Print the classification report
    report = classification_report(true_labels, pred)
    print(report)
    return report


def main_function():
    args = input_parse() # Command line arguments 
    unzip(args) # unzip the zip file
    folder_path_train, folder_path_test = delete() # Deleting folders
    datagen, test_datagen = image_generators() # Creating ImageDataGenerators
    training_tensorflow = data_generator(datagen, folder_path_train, "training") # Creating training images tensorflow
    validation_tensorflow = data_generator(datagen, folder_path_train, "validation") # Creating validation images tensorflow
    test_tensorflow = test_data_generator(test_datagen, folder_path_test) # Creating test images tensorflow
    model = pretrained_model() # Loading VGG16
    H = model_training(model, training_tensorflow, validation_tensorflow, args) # Training the model
    pred = predicting(test_tensorflow, training_tensorflow, model) # Testing the model with test images
    report = classifier_report(test_tensorflow, training_tensorflow, pred) # Creating classification report
    classifier_report_save(report, "synthetic_classification_report.txt") # Saving the report 
    plots(H, "synthetic_loss_and_accuracy_curve.png", args) # Creating plots
    model_save(model, "synthetic_model.keras") # Saving the model


if __name__ == "__main__": # If called from terminal run main function
    main_function()
