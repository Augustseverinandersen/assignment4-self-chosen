# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense,
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
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
    parser.add_argument("--zip_path", type=str, help = "Name of the zip folder")
    parser.add_argument("--train_sample_size", type = int, default = 75400, help = "Sample size of training images, default is all images")
    parser.add_argument("--epochs", type = int, default = 5, help = "How many epochs, default is 10")
    args = parser.parse_args()

    return args


## Unzipping
def unzip(args):
    folder_path_train = os.path.join("data", "asl_alphabet_train", "asl_alphabet_train") # Path to the data if unzipped already
    if not os.path.exists(folder_path_train): # Checking to see if folder is unzipped
        print("Unzipping file...")
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = os.path.join("data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination)
        print("The files are unzipped")
    return folder_path_train
   

## Deleting folders
def delete(folder_path_train):
    print("Deleting folders...")
    folders_to_delete = ["asl-alphabet-test", "del", "nothing", "space"]  # List of folders to delete

    for folder_name in folders_to_delete: # for every index in the above list
        folder_path = os.path.join(folder_path_train, folder_name) # Getting folder_path_train from unzip function 
        if os.path.exists(folder_path): # If the path exists
            shutil.rmtree(folder_path) # Delete the folder and its contents
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"Folder: {folder_path} is already deleted")


## Dataframe creation for train images 
def train_dataframe(folder_path_train):
    print("Creating dataframe of train images..")
    # Creating empty lists to store path and label
    image_paths = []
    labels = []

    # Finding the path to each letter folder, with os.listdir
    for label_folder in os.listdir(folder_path_train):
        letter_folder = os.path.join(folder_path_train, label_folder)

        # Finding path to each image in each folde.
        for filename in os.listdir(letter_folder):
            image_path = os.path.join(letter_folder, filename)
            
            # appedning path and folder name (label)
            image_paths.append(image_path)
            labels.append(label_folder)

    # Creating dataframe with two columns image_path and label
    train_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return train_df


## Creating a dataframe for test images from train dataframe 
def test_dataframe(train_df):
    print("Creating test dataframe")
    # Creating empty dataframe
    test_images_dataframe = pd.DataFrame(columns = ["image_path", "label"]) 
    # Grouping the dataframe by label
    labels_grouped = train_df.groupby("label") 
    # take the first 100 of each label and concatinates them into the new dataframe
    for label, group in labels_grouped: 
        test_images = group.head(100)
        test_images_dataframe = pd.concat([test_images_dataframe, test_images]) 
    # Removing the images with the same index from the train dataframe
    train_df = train_df.drop(test_images_dataframe.index) 
 
    return test_images_dataframe


## Sampling 
def sampling(train_df, args):
    print("Sampling")
    # Using Pandas sampling function to sample rows.
    train_df = train_df.sample(args.train_sample_size)
    return train_df


## Creating train and validation images 
def data_generator(datagen, train_df, train_or_val):
    print("Creating train and validation images")
    generator = datagen.flow_from_dataframe( # ImageDataGenerator
        train_df, # Train dataframe 
        x_col='image_path', # Path column
        y_col='label', # Label column
        target_size=(90, 90), # Load images as 90 by 90 pixels
        color_mode='rgb', # Colour images 
        class_mode='categorical', # One hot encode the labels
        batch_size=32, # Batch size of 32
        shuffle=True, # Shuffle the images 
        subset = train_or_val, # Subset will be either training or validation (Specified in mainfunction)
        interpolation='nearest') # Image resizing argument
    return generator


## Creating test images 
def test_data_generator(test_datagen, test_images_dataframe):
    print("Creating test images")
    test_tensorflow = test_datagen.flow_from_dataframe( # Test ImageDataGenerator 
        test_images_dataframe, # Test dataframe 
        x_col='image_path', # Path column
        target_size=(90, 90), # Load images as 90 by 90 pixels
        color_mode='rgb', # Colour images 
        class_mode = None, # No labels to encoded
        batch_size=32, # Batch size of 32
        shuffle=False) # Dont shuffle 
    return test_tensorflow


## Classifier report
def classifier_report(test_images_dataframe, pred):
    print("Creating classification report")
    true_labels = test_images_dataframe["label"].values # Get the labels from the dataframe
    report = classification_report(true_labels, pred) # Create the classification report with the labels and predictions
    print(report)
    return report


def main_function():
    args = input_parse() # Command line arguments
    folder_path_train = unzip(args) # Unzipping zip file
    delete(folder_path_train) # Deleting folders that are not letters
    train_df = train_dataframe(folder_path_train) # Creating a dataframe for all test images with labels and path
    test_images_dataframe = test_dataframe(train_df) # Creating a dataframe for test images from train dataframe
    train_df = sampling(train_df, args) # Sampling with argparse 
    datagen, test_datagen = image_generators() # ImageDataGenerator for train and validation
    training_tensorflow = data_generator(datagen, train_df, "training") # Train images with flow from dataframe
    validation_tensorflow = data_generator(datagen, train_df, "validation") # Validation images with flow from dataframe
    test_tensorflow = test_data_generator(test_datagen, test_images_dataframe)
    model = pretrained_model() # Loading VGG16
    H = model_training(model, training_tensorflow, validation_tensorflow, args) # Training the model with train images
    pred = predicting(test_tensorflow, training_tensorflow, model) # Testing the model with test images 
    report = classifier_report(test_images_dataframe, pred) # Creating the classification report 
    classifier_report_save(report, "real_classification_report.txt") # Saving the classification report 
    plots(H, "real_loss_and_accuracy_curve.png", args) # Creating plots and saving
    model_save(model, "real_model.keras") # Saving the model

if __name__ == "__main__": # If called from terminal run main function
    main_function()
