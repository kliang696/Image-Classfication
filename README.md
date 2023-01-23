# Image Classification with Deep Learning
### Introduction

The team is exploring the use of image recognition technology, specifically the application of image classification to animal detection. This technology can accurately identify different animal species, and can be applied in fields such as animal management, early childhood education, and image content analysis. The goal is to design a CNN model with 90% accuracy, using TensorFlow and exploring different CNN architectures and state-of-the-art pre-trained models. Data augmentation and experimental determination of hyperparameters such as learning rate, number of epochs, and batch size will also be used to improve the model's accuracy.

### Dataset Description
The team is going to use the “Animal Image Classification Dataset,” a source from Kaggle, which contains 12 classes of animals and at least 1,200 image files for each class.


#### Dataset Download
1. dirtecly download from kaggle: https://www.kaggle.com/datasets/piyushkumar18/animal-image-classification-dataset and then unzip the dataset
2. unzip dataset: 
- unzip animalclassification.zip
3. rename the dataset:
from pathlib import Path
DATA_DIR = os.getcwd() + os.path.sep + 'animalclassifcation' + os.path.sep
path = Path("D:\test")
temp = path.rename("Data")
new_path = temp.absolute()
## Coding part in Code folder
#### library required
import sklearn
import numpy
import tensorflow.keras
import seaborn
import matplotlib
import tensorflow
import pandas
from PIL import Image
### Dataset Splitting
train_test_split.py
### EDA
EDA_dataset.py
### Data Augmentation
data_augmentation.py
### Model ResNet50
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_ResNet50.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_ResNet50.py
### Model VGG19
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_VGG19.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_VGG19.py
### Model customized architeture
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_cus.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_cus.py
## Presentation in Presentation folder
check out our presentation
## Report in report folder
check out our report
