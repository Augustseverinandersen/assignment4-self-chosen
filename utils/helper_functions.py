import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

## Image datagenerator 
def image_generators():
    print("Creating Image data generator")
    # ImageDataGenerator from tensorflow 
    # Train and validation
    datagen = ImageDataGenerator(horizontal_flip=True, 
                                validation_split = 0.2, # Flip it horizontally around the access randomly 
                                 # Rotate the image randomly 20 degress around the access
                                rescale = 1/255) # rescale the pixel values to between 0-1                          
    # Test
    test_datagen = ImageDataGenerator(                 
                                rescale = 1./255.) # Datagenerator for test, it only has to rescale the images 
    return datagen, test_datagen 





# Loading VGG16
def pretrained_model():
    print("Loading model:")  
    # load model without classifier layers
    model = VGG16(include_top=False, # Exclude classifier layers
                pooling='avg',
                input_shape=(90, 90, 3)) # Input shape of the images. 224 pixels by 224. 3 color channels

    # Keep pretrained layers, and don't modify them
    for layer in model.layers:
        layer.trainable = False
        
    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) # Added batnormalization from tensorflow. Take the previouslayer, normalise the values, and than pass them on
    class1 = Dense(256, 
                activation='relu')(bn) # Added new classification layer 
    class2 = Dense(128, 
                activation='relu')(class1) # Added new classification layer with 15 outputs. 15 labels in total
    output = Dense(26, 
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, # Start learning rate at 0.01
        decay_steps=10000, # Every 10 000 steps start decaying 
        decay_rate=0.9) # DEcay by 0.9 to the start learning rate
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd, #Learning rate
                loss='categorical_crossentropy', # loss function
                metrics=['accuracy']) # What to display
    # summarize
    print(model.summary())
    return model





## Training
def model_training(model, training_tensorflow, validation_tensorflow, args):
    print("Training the model")
    H = model.fit( # fitting the model to 
        training_tensorflow, # training data from tensorflow dataframe 
        steps_per_epoch = len(training_tensorflow), # Take as many steps as the length of the dataframe 
        validation_data = validation_tensorflow, # Validation data from tensorflow dataframe
        validation_steps = len(validation_tensorflow), # Validation steps as length of validation data 
        epochs = args.epochs)
    return H





## Predictions
def predicting(test_tensorflow, training_tensorflow, model):
    print("Testing the model")
    pred = model.predict(test_tensorflow) # Using test data on the model
    pred = np.argmax(pred,axis=1) # Convert to highest label

    # Mapping the labels
    labels = (training_tensorflow.class_indices) 
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    return pred





# Saving classification report 
def classifier_report_save(report, name): # Saving classification report
    print("Saving classification report")
    folder_path = os.path.join("out") # Saving in folder models
    file_name = name # Name of the file
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(report)
    print("Reports saved")




## Creating plots and saving them
def plots(H, name, args):
    print("Creating plots")    
    # Plotting the loss
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(np.arange(0, args.epochs), H.history["loss"], label="train_loss") #plotting loss
    plt.plot(np.arange(0, args.epochs), H.history["val_loss"], label="val_loss", linestyle=":") # plotting validation loss
    plt.xlabel("Epoch") # x axis label
    plt.ylabel("Loss") # y axis label
    plt.title("Training and Validation Loss") # title 
    plt.legend() # legend 

    # Plotting the accuracy
    plt.subplot(1, 2, 2) # 2nd plot
    plt.plot(np.arange(0, args.epochs), H.history["accuracy"], label="accuracy") # plotting accuracy
    plt.plot(np.arange(0, args.epochs), H.history["val_accuracy"], label="val_accuracy", linestyle=":") # plotting validation accuracy
    plt.xlabel("Epoch") # x axis label
    plt.ylabel("Accuracy") # y axis label
    plt.title("Training and Validation Accuracy") # title
    plt.legend() # legend

    # Adjusting the layout and saving the plots
    plt.tight_layout() # layout
    plt.savefig(os.path.join("figs", name)) # saving the plot





## Saving model
def model_save(model, name):
    print("Saving model")
    folder_path = os.path.join("models", name) # Defining out path
    tf.keras.models.save_model( # Using Tensor Flows function for saving models.
    model, folder_path, overwrite=False, save_format=None 
    ) # Model name, folder, Overwrite existing saves, save format = none 

