# 4. Assignment4-ASL-image-classification 
## 4.1 Assignment Description 
This is a self chosen assignment. I have chosen to work with American Sign Language alphabet images (ASL). This assignment will use a pre-trained model to classify to different datasets of ASL, create a classification report for both models, save a loss and accuracy curve plot, and save the models.
## 4.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Python version 1.73.1. The script ```asl_real.py``` took 80 minutes to run with 32-CPU, with the majority of the time spent on training the model. The script ```asl_synthetic.py``` took 100 minutes to run with 32-CPU. 
### 4.2.1 Prerequisites 
To run this script, make sure to have Bash and Python 3 installed on your device. This script has only been tested on Ucloud. 
## 4.3 Contribution 
This assignment takes inspiration from __Assignment3-Pretrained CNNs-Using pretrained CNNs for image classification__, and from [Vijayabhaskar J.](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c), who has made a guide about using TensorFlows *flow_from_dataframe*. This assignment uses the pre-trained convolutional nerual network *VGG16*, created by [K. Simonyan and A. Zisserman](https://neurohive.io/en/popular-networks/vgg16/). Furthermore, the data used in this assignment is from Kaggle. The synthetic data is made by [Lexset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) and the real data is made by [AKASH](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) 
### 4.3.1 Data
__The synthetic dataset__ consits of 27000 images of the alphabet signed in American Sign Language. The images are originally 512 by 512 pixels. The structure of the data is as follows: Two folders (Test_Alphabet, Train_Alphabet), each containing 27 folders (one for each letter of the alphabet, and a blank folder). Each letter folder in the train folder has 900 images, and in the test folder has 100 images. The images were created using the software tool [Lexset](https://www.lexset.ai/). Lexset is a synthetic data company, that uses artificial intelligence to create photorealistic synthetic data [source](https://www.linkedin.com/company/lexset/). The ASL data is created using 3D models of ASL letters, and appling synthetic backgrounds and colour to each image, thereby creating a various dataset.

__The real dataset__ consits of 87000 images of 200 by 200 pixels. The structure of the data is as follows: Two folders (asl_alphabet_test, and asl_alphabet_train), each containing a sub folder of the same name, containing 29 folders (one for each letter of the alphabet, and three additional classes *SPACE, DELETE,* and *NOTHING*), containing the images. The train folders contain 3000 images each, and the test folders contain one image each. The data appears to be created by one person because he "*was inspired to create the dataset to help solve the real-world problem of sign language recognition.*". The background of the images is almost the same, the hand placement various abit, but the images have different lighting.

## 4.4 Packages
This script uses the following packages:
- TensorFlow (version 2.12.0) the following is beimg imported: *ImageDataGenerator* is used to load and augment the images. *VGG16* is the pretrained CNN. *Flatten, Dense, and BatchNormalization* is used to create new classifier layers. *ExponentialDecay and SGD* is used to create a learning decay rate.
- Scikit-learn (version 1.2.2) is used to import the *classification report*.
- Numpy (version 1.23.5) is used to handle arrays.
- Shutil is used to delete folders with content.
- Pandas (version 2.0.1) is used to create the dataframes and sampling.
- Matplotlib (3.7.1) is used to create plots.
- Os is used to navigate file paths, on different operating systems.
- Zipfile is used to unpack the zip files.
- Argparse is used to create command line arguments
- Sys is used to navigate the directory. In this case show the pack to the ```helper_functions.py``` script.
## 4.5 Repository Contents
This repository contains the following folder and files.
- ***Data*** is an empty folder where the zip files will be placed.
- ***Figs*** is a folder that contains the *loss and accuracy plots*. 
- ***Models*** is a folder that contains the saved models. 
- ***Out*** is a folder that contains the classification reports for both models.
- ***Src*** is a folder that contains the two scripts ```asl_real.py``` and ```asl_synthetic.py```.
- ***Utils*** is a folder that contains the ```helper_functions.py``` script. 
- ***README.md*** the readme file.
- ***Requirements.txt*** this text file contains the packages to install.
- ***Setup.sh*** installs the virtual environment, upgrades pip, and installs the packages from requirements.txt.
## 4.6 Methods
### 4.6.1 Script ```asl_synthetic.py```
- The script starts by initialising command-line arguments for the path to the zip file and epochs. 
- Then, it unzips the zip file, if it is not unzipped.
- The script than deletes the folder *blank*, using *rmtree* from the *shutil* package. This is done becuase I only want the scripts to classify ASL letters.
- The script than takes the function ```image_generator``` from the ```helper_functions.py``` in the utils folder. This function uses TensorFlows *ImageDataGenerator*, which loads and augments images. Two data generators are created. The first one is used on the train and validation images, here some of the images are flipped horizontally, a validation split of 20 % is created, and the images are rescaled to 0-1 (easier to compute). The second data generator is for the test images, the only thing that happens to them is rescaling to 0-1.
- With the *ImageDataGenerators* created, the training images are loaded in using TensorFlows *flow_from_directory*. The images are loaded in as batches of 32 colour images by size 90x90 pixels, the labels are one-hot encoded, the images are shuffled, a subset for either *training* or *validation* is created, and the rescaling function *nearest* is chosen. The same goes for the validation images, excpet that the subset is *validation* which takes 20% of the training images, as specified in the ImageDataGenerator.
- The test images are loaded in using *flow_from_directory* aswell. They are loaded in at batches of 32, as colour images of pixel size 90x90. Here shuffeling is set as false, no label is specified, but the rescaling function is the same.
- The train images, validation images, and test images are stored as tuples of data containing two arrays of input features and labels. 
- From the ```helper_functions-py``` script, the pretrained model, VGG16, is loaded. It is loaded with the following arguments:
  - Image size 90x90 pixels.
  - Colour images.
  - Without classifier layers.
  - The weights of the pretrained layers unchangeable.
  - New classifier layers consisting of a flatten layer, a batch normalization layer, two layers with relu activation, and a softmax layer to make the output of the previous layer into a probability distribution of the 26 letters in the Alphabet. 
- From the ```helper_functions-py``` script, the function, ```model_training``` for training the model is loaded. This function takes the training images and the validation images and has the amount of epochs as an argparse, that can be specified by you. 
- From the ```helper_functions-py``` script, the function, ```predicting```, for predicting is loaded. This function test the model on the test images, and stores the prediction for the label with the highest value. This function also maps the labels from their numerical value to their string value.
- A classification report is than created, by getting the true labels and the predicted labels and comparing them.
- Lastly, From the ```helper_functions-py``` script, the functions for saving the classifier, plotting (and saving) the loss and accuracy curve, and for saving the model is created. The classification report is saved in the folder *out*, the plot is saved in the folder *figs*, and the model is saved in the folder *models*
### 4.6.2 Script ```asl_real.py``` 
This script uses the same functions from the ```helper_functions.py``` as the previous script, so they will not be explained again.
- The script starts by making initialising command-line arguments for the path to the zip file, sample size, and amount of epochs.
- Then the scripts deletes the folders *del, nothing,* and *space*. To ensure that only letters are included in the image classification.
- After deleting, the script creates a Pandas dataframe of all train images with two columns, *image_path* and *label*. This dataframe is created by using for loops and *os.listdir* to get the filepath, and each folder name (labels). The reason a dataframe is created is to allow for sampling, and to get test images. Since the dataset only has one test image per class.
- The test image dataframe is than created by grouping the dataframe together by the column label, and getting the first 100 rows for each label. Lastly, the rows that have been moved to the new dataframe are deleted in the original dataframe, to ensure no duplicates.
- Than a function for sampling is created. This function takes an argparse as the amount of rows to be sampled. Sampling is benefical to reduce the training size, and thereby reducing training time. I chose to create a sample of 26000 images out of the 75400 possible training images. Out of the 26000 iamges 5200 of them became validation images.
- The train and validation images are than loaded in using TensorFlows *flow_from_dataframe*. The *flow_from_dataframe* uses the same arguments as the *flow_from_directory*, except that the path to the images and the labels are from columns.
- The test images are than loaded in the same way as the train and validation images, with the exception of shuffle is false, and no labels are specified.
- After training the model, from the ```helper_functions.py``` script, a classification report is created by getting the true labels from the column labels, and comparing them to the predictions. 
- Lastly, the model, plot, and classification report is saved.

## 4.7 Discussion
Looking at the classification report for both scripts, shows that both models perform well considering that there are 26 classes to predict. However, one of the models has a significantly higher score then the other. The ```asl_real.py```script had an accuracy f1-score of 0.94 compared to ```asl_synthetic.py``` that had an accuracy f1-score of 0.54. Both models were tested on 2600 images (100 of each letter). For the synthetic script, the worst f1-score was for the letter *F* (0.41), which was one of the letters with the best f1-score on the real script (0.99). The reason that the ```asl_real.py``` script performs better than the other script is that there are no big variations in background features. The real script only had bland background, meaning that the model could purely learn the features of the hand. The ```asl_synthetic.py```script had a lot of various backgrounds with different levels of features. This impacted the models learning, since the model thinks that all features in the image are part of the letter.

When looking at the *loss and accuracy plotss* for the two scripts, it clearly shows that one model is not over or under fitting, but one model is over fitting. The ```asl_real.py```script follows a steady curve for both training and validation. This shows that the model is not over fitting and that it is not under fitting as well. The plot for the ```asl_synthetic.py``` script shows that the model is either over fitting or that the images are to different for the model to learn weights for good predictions. The *train and validation loss* shows that the *val_loss* curve is cutting through the *train_loss* curve, and that it is not flattining out during the end of training, which are signs of over fitting. The same goes for the *training and validation accuracy*.
One of the main factors in the difference in model performance is that the images have different background features, as discussed previously. However, an other issues is that the ```asl_synthetic.py``` images were rescaled from 500x500 pixels to 90x90 pixels. This significant downscaling saves computation power and time, but could influence what features are extracted from the image, since down scaling could removes small significat features. 

It would have been interesting to see how the model would have performed if the images were rescaled to half of the original size, but due to computational limitations this was not possible. Furthermore it would have been interesting to see how the ```asl_real.py``` script would have performed on all training images instead of a subset. Lastly, it would have been interesting to see how the models would have performed on images from the other script (Synthetic model on real images)(Real model on synthetic images). My hypothesis is that both models would have performed badly, since they have been trained on a hugh difference in amount of features to predict on.

## 4.8 Usage
To run the scripts in this repository follow these steps:
1. Clone the repository.
2. Get the zip file data from Kaggle for the [synthetic images](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) and the [real images](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), and place them in the *data* folder.
3. Run ```bash setup.sh``` in the command-line. This will create a virtual environment and install the requirements.
4. Run ```source ./assignment_4/bin/activate``` in the command-line to activate the virutal environment.
5. To run the ```asl_real.py``` script, run this in the command-line: ```python3 src/asl_real.py --zip_path data/archive.zip --train_sample_size 26000 --epochs 5```
  - ```--zip_path``` takes a string as input and is the path to your zip file. 
  - ```--train_sample_size``` takes an integer as input, but has a defualt of 75400. Only include it if you want to sample.
  - ```--epochs``` takes an integer as input, but has a defualt of 5. Only include if you want to change how many epochs the model will be trained on.
6. To run the ```asl_synthetic.py``` script, run this in the command-line: ```python3 src/asl_synthetic.py --zip_path data/archive1.zip --epochs 10```
  - ```--zip_path``` takes a string as input and is the path to your zip file.
  - ```--epochs``` takes an integer as input, but has a defualt of 10. Only include if you want to change how many epochs the model will be trained on.
