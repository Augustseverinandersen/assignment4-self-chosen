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
The synthetic dataset consits of 27000 images of the alphabet signed in American Sign Language. The images are originally 512 by 512 pixels. The structure of the data is as follows: Two folders (Test_Alphabet, Train_Alphabet), each containing 27 folders (one for each letter of the alphabet, and a blank folder). Each letter folder in the train folder has 900 images, and in the test folder has 100 images. The images were created using the software tool [Lexset](https://www.lexset.ai/). Lexset is a synthetic data company, that uses artificial intelligence to create photorealistic synthetic data [source](https://www.linkedin.com/company/lexset/). The ASL data is created using 3D models of ASL letters, and appling synthetic backgrounds and colour to each image, thereby creating a various dataset.

The real dataset consits of 87000 images of 200 by 200 pixels. The structure of the data is as follows: Two folders (asl_alphabet_test, and asl_alphabet_train), each containing a sub folder of the same name, containing 29 folders (one for each letter of the alphabet, and three additional classes *SPACE, DELETE,* and *NOTHING*), containing the images. The train folders contain 3000 images each, and the test folders contain one image each. The data appears to be created by one person because he "*was inspired to create the dataset to help solve the real-world problem of sign language recognition.*". The background of the images is almost the same, the hand placement various abit, but the images have different lighting.
## 4.4 Packages
## 4.5 Repository Contents
## 4.6 Methods
### 4.6.1 Script 
### 4.6.2 Script
## 4.7 Discussion
discus pixel downsizing
## 4.8 Usage
