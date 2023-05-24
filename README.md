# 4. Assignment4-ASL-image-classification 
## 4.1 Assignment Description 
This is a self-chosen assignment. I have chosen to work with American Sign Language (ASL) alphabet images. This assignment will use a pre-trained model to classify two different datasets of ASL, create a classification report for both models, save a loss and accuracy curve plot, and save the models.
## 4.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. This script ran on Coder Python 1.73.1 and Python version 3.9.2. The script ``asl_real.py`` took 80 minutes to run with 32-CPU, with the majority of the time spent on training the model. The script ``asl_synthetic.py`` took 100 minutes to run with 32-CPU.
### 4.2.1 Prerequisites 
To run this script, make sure to have Bash and Python 3 installed on your device. This script has only been tested on Ucloud. 
## 4.3 Contribution 
This assignment takes inspiration from **Assignment3-Pretrained CNNs-Using pretrained CNNs for image classification**, and from [Vijayabhaskar J.](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c), who has made a guide about using TensorFlows _flow_from_dataframe_. This assignment uses the _pre-trained convolutional neural network VGG16_, created by [K. Simonyan and A. Zisserman](https://neurohive.io/en/popular-networks/vgg16/). Furthermore, the data used in this assignment is from Kaggle. The synthetic data is made by [Lexset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) and the real data is made by [AKASH](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
### 4.3.1 Data
**The synthetic dataset** consists of 27000 images of the alphabet signed in American Sign Language. The images are originally 512 by 512 pixels. The structure of the data is as follows: Two folders (Test_Alphabet, Train_Alphabet), each containing 27 folders (one for each letter of the alphabet, and a _blank_ folder). Each letter folder in the train folder has 900 images, and in the test folder, each letter folder has 100 images. The images were created using the software tool [Lexset](https://www.lexset.ai/). Lexset is a synthetic data company, that uses artificial intelligence to create photorealistic synthetic data - [source](https://www.linkedin.com/company/lexset/). The ASL data is created using 3D models of ASL letters and applying synthetic backgrounds and colour to each image, thereby creating a dataset with various images.

**The real dataset** consists of 87000 images of 200 by 200 pixels. The structure of the data is as follows: Two folders (asl_alphabet_test, and asl_alphabet_train), each containing a subfolder of the same name, containing 29 folders (one for each letter of the alphabet, and three additional classes _SPACE, DELETE_, and _NOTHING_), containing the images. The train folders contain 3000 images each, and the test folders contain one image each. The data appears to be created by one person because he _"was inspired to create the dataset to help solve the real-world problem of sign language recognition."_. The background of the images is almost the same, but the hand placement is not always in the center of the image. Furthermore, the images are taken with different lighting.
## 4.4 Packages
This script uses the following packages:
-	TensorFlow (version 2.12.0) the following is being imported: _ImageDataGenerator_ is used to load and augment the images. _VGG16_ is the pre-trained CNN. _Flatten, Dense, and BatchNormalization_ is used to create new classifier layers. _ExponentialDecay_ and _SGD_ are used to create a learning decay rate.
-	Scikit-learn (version 1.2.2) is used to import the _classification report_.
-	Numpy (version 1.23.5) is used to handle arrays.
-	Shutil is used to delete folders with content.
-	Pandas (version 2.0.1) is used to create the dataframes and sampling.
-	Matplotlib (3.7.1) is used to create plots.
-	Os is used to navigate file paths, on different operating systems.
-	Zipfile is used to unpack the zip files.
-	Argparse is used to create command line arguments
-	Sys is used to navigate the directory. In this case show the path to the helper_functions.py script.
## 4.5 Repository Contents
This repository contains the following folder and files.
- ***Data*** is an empty folder where the zip files will be placed.
- ***Figs*** is a folder that contains the *loss and accuracy plots*. 
- ***Models*** is a folder that contains the saved models. 
- ***Out*** is a folder that contains the _classification reports_ for both models.
- ***Src*** is a folder that contains the two scripts ```asl_real.py``` and ```asl_synthetic.py```.
- ***Utils*** is a folder that contains the ```helper_functions.py``` script. 
- ***README.md*** the readme file.
- ***Requirements.txt*** this text file contains the packages to install.
- ***Setup.sh*** installs the virtual environment, upgrades pip, and installs the packages from requirements.txt.
## 4.6 Methods
### 4.6.1 Script ```asl_synthetic.py```
-	The script starts by initializing command-line arguments for the path to the zip file and epochs.
-	Then, it unzips the zip file, if it is not unzipped.
-	The script then deletes the folder blank, using _rmtree_ from the shutil package. This is done because I only want the scripts to classify ASL letters.
-	The script then takes the function _image_generator_ from the ``helper_functions.py`` in the utils folder. This function uses TensorFlows _ImageDataGenerator_, which loads and augments images. Two data generators are created. The first one is used on the train and validation images, here some of the images are flipped horizontally, a validation split of 20% is created, and the images are rescaled to 0-1 (easier to compute). The second data generator is for the test images, the only thing that happens to them is rescaling to 0-1.
-	With the _ImageDataGenerators_ created, the training images are loaded in using TensorFlows _flow_from_directory_. The images are loaded in batches of 32 colour images of 90x90 pixels, the labels are one-hot encoded, the images are shuffled, a subset for either training or validation is created, and the rescaling function nearest is chosen. The same goes for the validation images, except that the subset is validation which takes 20% of the training images, as specified in the _ImageDataGenerator_.
-	The test images are loaded using _flow_from_directory_ as well. They are loaded in batches of 32, as colour images of pixel size 90x90. Here shuffling is set as false, and no label is specified, but the rescaling function is the same.
-	The train images, validation images, and test images are stored as tuples of data containing two arrays of input features and labels.
-	From the ``helper_functions.py`` script, the pre-trained model, _VGG16_, is loaded. It is loaded with the following arguments:
    - Image size 90x90 pixels.
    - Colour images.
    - Without classifier layers.
    - The weights of the pre-trained layers are unchangeable.
    - New classifier layers are created consisting of a _flatten layer_, a _batch normalization layer_, _two layers with relu activation_, and a _softmax layer_ to make the output of the previous layer into a probability distribution of the 26 letters in the alphabet.
    - From the ``helper_functions.py`` script the function, ``model_training``, for training the model is loaded. This function takes the training images and the validation images and has the amount of epochs as an argparse, that can be specified by you. The model is then trained, and progress is printed to the command line.
    - From the ``helper_functions.py`` script the function, ``predicting``, for predicting is loaded. This function tests the model on the test images and stores the prediction for the label with the highest value. This function also maps the labels from their numerical value to their string value.
    - A classification report is then created, by getting the true labels and the predicted labels and comparing them.
- Lastly, from the ``helper_functions.py`` script, the functions for saving the classifier, plotting (and saving) the loss and accuracy curve, and saving the model are created. The classification report is saved in the folder _out_, the plot is saved in the folder _figs_, and the model is saved in the folder _models_
### 4.6.2 Script ```asl_real.py``` 
This script uses the same functions from the ``helper_functions.py`` as the previous script, so they will not be explained again.
-	The script starts by initializing command line arguments for the path to the _zip file, sample size_, and amount of _epochs_.
-	Then the script deletes the folders _del_, _nothing_, and _space_. To ensure that only letters are included in the image classification.
-	After deleting, the script creates a Pandas data frame of all train images with two columns, _image_path_ and _label_. This data frame is created by using loops and _os.listdir_ to get the file path, and each folder name (labels). The reason a data frame is created is to allow for sampling and to create test images. Since the dataset only has one test image per class.
-	The test image data frame is then created by grouping the data frame by the column label and getting the first 100 rows for each label. Lastly, the rows that have been moved to the new data frame are deleted in the original data frame, to ensure no duplicates.
-	Then a function for sampling is created. This function takes an argparse as the number of rows to be sampled. Sampling is beneficial to reduce the training size, thereby reducing training time. I chose to create a sample of 26000 images out of the 75400 possible training images. Out of the 26000 images, 5200 of them became validation images.
-	The train and validation images are then loaded using TensorFlows _flow_from_dataframe_. The _flow_from_dataframe_ uses the same arguments as the _flow_from_directory_, except that the path to the images and the labels are from columns.
-	The test images are then loaded in the same way as the train and validation images, except _shuffle_ is _false_, and no labels are specified.
-	After training the model, from the ``helper_functions.py`` script, a classification report is created by getting the true labels from the column label and comparing them to the predictions.
-	Lastly, the model, plot, and classification report are saved.


## 4.7 Discussion
### 4.7.1 Performance 
Looking at the classification report for both scripts shows that both models perform well considering that there are 26 classes to predict. However, one of the models has a significantly higher score than the other. The ``asl_real.py`` script had an _accuracy f1-score_ of 0.94 compared to ``asl_synthetic.py`` which had an _accuracy f1-score_ of 0.54. Both models were tested on 2600 images (100 of each letter). For the synthetic script, the worst _f1-score_ was for the letter F (0.41), which was one of the letters with the best _f1-score_ in the real script (0.99). The reason that the ``asl_real.py`` script performs better than the other script is that there are no big variations in background features. The real script only had bland backgrounds, meaning that the model could purely learn the features of the hand. The ``asl_synthetic.py`` script had a lot of various backgrounds with different levels of features. This impacted the model’s learning since the model thinks that all features in the image were part of the letter.
### 4.7.2 Loss and Accuracy Plots
When looking at the _loss and accuracy plots_ for the two scripts, it clearly shows that one model is not over or underfitting and that one model is overfitting. The ``asl_real.py`` script follows a steady curve for both training and validation. This shows that the model is not overfitting and that it is not underfitting as well. The plot for the ``asl_synthetic.py`` script shows that the model is either overfitting or that the images are too different for the model to learn weights for good predictions. The _train and validation loss_ shows that the _val_loss_ curve is cutting through the _train_loss_ curve and that it is not flattening out during the end of the training, which are signs of overfitting. The same goes for the _training and validation accuracy_. One of the main factors in the difference in model performance is that the images have different background features, as discussed previously. However, another issue is that the synthetic images were rescaled from 500x500 pixels to 90x90 pixels. This significant downscaling saves computation power and time but could influence what features are extracted from the image since downscaling could remove small significant features.
### 4.7.3 Future Analysis and Considerations 
It would have been interesting to see how the model would have performed if the images were rescaled to half of the original size, but due to computational limitations, this was not possible. Furthermore, it would have been interesting to see how the ``asl_real.py`` script would have performed on all training images instead of a subset. Lastly, it would have been interesting to see how the models would have performed on images from the other script (_Synthetic model on real images_)(_Real model on synthetic images_). I hypothesize that both models would have performed badly since they have been trained on a huge difference in the amount of features to predict on.


## 4.8 Usage
To run the scripts in this repository follow these steps:
-	Clone the repository.
-	Navigate to the correct directory.
-	Get the zip file data from Kaggle for the [synthetic images](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) and the [real images](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), and place them in the _data_ folder (you might need to rename them).
-	Run ``bash setup.sh`` in the command line. This will create a virtual environment and install the requirements.
-	Run ``source ./assignment_4/bin/activate`` in the command line to activate the virtual environment.
-	To run the ``asl_real.py script``, run this in the command line: 
``python3 src/asl_real.py --zip_path data/archive.zip --train_sample_size 26000 --epochs 5``
    - ``--zip_path`` takes a string as input and is the path to your zip file.
    - ``--train_sample_size`` takes an integer as input but has a **default of 75400**. Only include it if you want to sample.
    - ``--epochs`` takes an integer as input but has a **default of 5**. Only include if you want to change how many epochs the model will be trained on.
-	To run the ``asl_synthetic.py`` script, run this in the command line: 
``python3 src/asl_synthetic.py --zip_path data/archive1.zip --epochs 10``
    - ``--zip_path`` takes a string as input and is the path to your zip file.
    - ``--epochs`` takes an integer as input but has a **default of 10**. Only include if you want to change how many epochs the model will be trained on.

