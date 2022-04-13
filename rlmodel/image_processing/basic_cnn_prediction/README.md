# Basic CNN and evaluation
This directory contains saved models, source code, and notebook for basic convolutional neural network (CNN) for cylinder position extraction using image. 

## Basic_CNN
`Basic_CNN.ipynb` trains a CNN network which contains two CNN layers and two fully connected (FC) layers. The CNN layers are first trained on the CIFAR-10 dataset and then trained on our custom dataset that is generated from the data generator (under `image_processing/src/data_generator`). 

## CNN_Agent
`CNN_Agent.ipynb` uses our best TD3 4 joints reinforcement learning (RL) model to evaluate the performance of the CNN model. 

## saved_models
Under the directory `saved_models`, there are two models saved from training and used for evaluation. `cnn_model` is the saved model of CNN. `pretrain_model_params` is the saved model parameters of the pretrained model. Their usage can be seen from the `Basic_CNN.ipynb`. 

## src
`src` directory contains necessary code for CNN model training and evaluation. 