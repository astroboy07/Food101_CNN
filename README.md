![cover_image](https://github.com/astroboy07/Food101_CNN/blob/main/Extras/food-101.jpg)

# Food Recognition 🍕🔎

## Overview

This is an end-to-end machine learning model for recognizing food in your images! The model is trained on 101 classes of food ranging from ramen to chocolate cake to chicken curry to edamame. All right, you get the idea!


## What?

The is basically an image classification model built using TensorFlow/Keras and deployed using Streamlit. The main directories in this repositor are:

1. [Notebooks](https://github.com/astroboy07/Food101_CNN/tree/main/Notebooks): contains the python notebook file where the model is built, trained, and tested.
2. [Saved_model](https://github.com/astroboy07/Food101_CNN/tree/main/Saved_model): contains the best model saved in `hdf5` format.
3. [Utils](https://github.com/astroboy07/Food101_CNN/tree/main/Utils): contains the python script used to deply the model using Streamlit.


## Goal

The aim of this project is to beat the accuracy of **77.4%** achieved by the [paper](https://arxiv.org/abs/1606.05675) where they trained the model for **2-3 days**. Bonus point will be given if we can reduce the training time significantly.

## How?

### Data

The dataset used for training the model is obtained from [`tfds.image_classification.Food101`](https://www.tensorflow.org/datasets/catalog/food101). This dataset has 101,000 images of 101 different classes of food. The dataset is split into training and test (validation) dataset with 75,750 and 25,250 images respectively. 

### Training

We optimized the data input pipeline and used mixed precision training to achieve peak performance by using compatible GPU. Firstly, we employed varoious techniques such as prefetch transformation, parallelization of map transformation etc. ([more details](https://www.tensorflow.org/guide/data_performance#best_practice_summary)) to achieve efficient input data pipeline. Secondly, we used [`mixed precision`](https://www.tensorflow.org/guide/mixed_precision) the uses both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. This framework of training increased the computation speed by 3x.

### Model

We start off by building a feature extraction model with a base model and customized input and output layer satisfying our needs. The base model is [`EfficientNetB1`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB1) which has been originally trained on ImageNet dataset. We use the power of transfer learning and fine-tuning to achieve accuracy of **80%**. We also got the bonus point because our model training was over in a couple of minutes compared to the mentioned **2-3** days time in the paper.

## Evaluation

We evaluate the model using various metrics on the "unseen" test data. For more details, please refer to [`Food_Vision.ipynb`](https://github.com/astroboy07/Food101_CNN/blob/main/Notebooks/Food_Vision.ipynb). 

## Deployment

Finally, we deployed the model using the Streamlit. The python script used can be found [here](https://github.com/astroboy07/Food101_CNN/blob/main/Utils/app.py). This part is directly inspired by **Daniel Bourke's** [CS329s Lecture](https://www.mrdbourke.com/cs329s-machine-learning-deployment-tutorial/). 
