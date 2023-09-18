# Machine_Vision-Carbon_Nanomaterial_Identification
Model testing for Carbon Nanomaterial Phase Identification using Transfer Learning and Machine Vision.

## Overview

This repository contains code and resources for the model presented in the paper "Leveraging Transfer Learning and Machine Vision for Enhanced Carbon Nanomaterial Phase Identification with Scanning Electron Microscopy." The model is designed to identify different carbon nanomaterial phases using transfer learning and convolutional neural networks (CNNs) based on scanning electron microscopy (SEM) images.

## Abstract

In this report, we propose a novel technique for the identification and analysis of various nanoscale
carbon structures using scanning electron microscopy. Through precise control of quenching rates,
achieved by laser irradiation of undercooled molten carbon, we successfully formed
microdiamonds, nanodiamonds, and Q-carbon films. However, standard laser irradiation without
proper control leads to the formation of different carbon polymorphs, making their classification
challenging through manual analysis. To address this issue, we applied transfer-learning
approaches using convolutional neural networks and computer vision techniques. We provide
comprehensive insights by performing the analysis of five transfer-learning models: VGG16,
ResNet50, MobileNetV2, DenseNet169, and InceptionV3, on six carbon materials phases and
substrates. Our method achieved high accuracy rates of 91% overall accuracy on the DenseNet169
model, for identifying Q-carbon and 94% for distinguishing it from nanodiamonds. Remarkably,
it was able to distinguish Q-carbon and nanodiamonds ensemble, CVD diamonds, graphene,
carbon nanotubes, and substrates with 100% accuracy. By leveraging SEM images and precise
undercooling control, our technique enables efficient identification and characterization of
nanoscale carbon structures. This research significantly contributes to the field by providing
automated tools for Q-material and carbon polymorph identification, opening new opportunities
for their exploration in various applications.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [VGG16](#vgg16)
  - [ResNet50](#resnet50)
  - [MobileNetV2](#mobilenetv2)
  - [DenseNet169](#densenet169)
  - [InceptionV3](#inceptionv3)
- [Testing](#testing)
- [License](#license)

## Usage

Researchers and practitioners can utilize the model in this repository to perform phase identification of carbon nanomaterials from SEM images. The provided code demonstrates how to load the pre-trained model, preprocess SEM images, and obtain accurate predictions for different carbon phases.

## Dependencies

To run the Carbon Nanomaterial Phase Identification model and associated code, you need to have the following libraries and modules installed:

- requests
- tensorflow
- matplotlib.image
- numpy
- collections.defaultdict
- collections
- shutil
- tensorflow.keras.backend
- tensorflow.keras.models.load_model
- tensorflow.keras.preprocessing.image
- matplotlib.pyplot
- tensorflow
- tensorflow.keras
- tensorflow.keras.layers
- tensorflow.keras.callbacks
- tensorflow.keras.optimizers
- tensorflow.keras.regularizers
- cv2
- tensorflow

You can install these required packages using pip:

```bash
pip install requests tensorflow matplotlib numpy opencv-python
```

## Usage
Researchers and practitioners can utilize the model in this repository to perform phase identification of carbon nanomaterials from SEM images. The provided code demonstrates how to load the pre-trained model, preprocess SEM images, and obtain accurate predictions for different carbon phases.

## Dataset
The dataset used to train and evaluate the model is not included in this repository due to size constraints. However, you can obtain the dataset from autrhors.. Make sure to organize the dataset into appropriate directories (e.g., train, test) before running the model.

## Model Architecture

### VGG16

The VGG16 model, known for its deep and straightforward architecture, plays a pivotal role in our project. This model, pre-trained on the ImageNet dataset, serves as a strong feature extractor. In our adaptation, we retain the original VGG16 architecture up to the final convolutional block.

We augment VGG16 with custom layers for fine-tuning on the carbon nanomaterial phase identification task. These additional layers are designed to adapt the model to the nuances of our specific domain. We incorporate a global average pooling layer followed by a fully connected layer with 128 neurons and a ReLU activation function. Dropout with a rate of 20% is applied to reduce overfitting.

The final classification layer employs a softmax activation function, allowing the model to make multi-class predictions efficiently. To mitigate the risk of overfitting and enhance generalization, a kernel regularization term with L2 regularization strength of 0.005 is added.

### ResNet50

ResNet50 is another cornerstone of our project's model architecture. Renowned for its deep residual learning, this model has the capability to capture intricate features from images. Our adaptation retains the ResNet50 architecture, excluding the final classification layer.

To tailor ResNet50 for our carbon nanomaterial phase identification task, we append custom layers. These additional layers include global average pooling followed by a densely connected layer with 128 units and a ReLU activation function. A dropout layer with a 20% dropout rate is applied to prevent overfitting.

The classification layer at the end employs a softmax activation function, enabling the model to perform multi-class classification. Regularization is incorporated with an L2 regularization term of 0.005 to enhance model generalization.

### MobileNetV2

MobileNetV2 is selected for its efficiency and effectiveness in extracting features from images. In our model adaptation, we preserve the MobileNetV2 architecture up to the penultimate layer.

Custom layers are added to fine-tune MobileNetV2 for the carbon nanomaterial phase identification task. These layers consist of global average pooling, followed by a dense layer with 128 units and ReLU activation. To prevent overfitting, a dropout layer with a 20% dropout rate is introduced.

The final classification layer utilizes a softmax activation function for multi-class predictions. Regularization in the form of L2 regularization with a strength of 0.005 enhances the model's robustness.

### DenseNet169

DenseNet169, known for its densely connected layers, is another integral part of our model architecture. We retain the original DenseNet169 architecture while removing the final classification layer.

Custom layers are introduced to adapt DenseNet169 for the carbon nanomaterial phase identification task. These include global average pooling followed by a dense layer with 128 units and ReLU activation. A dropout layer with a 20% dropout rate is employed to mitigate overfitting.

The final classification layer utilizes a softmax activation function for multi-class predictions. L2 regularization with a strength of 0.005 is applied to improve the model's generalization capacity.

### InceptionV3

The core of this project's success lies in the model architecture based on transfer learning with the InceptionV3 model. InceptionV3 is a powerful CNN architecture known for its ability to extract intricate features from images, making it ideal for various image recognition tasks.

The proposed model utilizes the InceptionV3 model, but the last layer specific to ImageNet classification is removed. Instead, we add new layers to the model for fine-tuning on the carbon nanomaterial phase identification task. The model is then compiled with the stochastic gradient descent (SGD) optimizer and categorical cross-entropy loss function for effective multi-class classification.


## Training the NN Algorithm
The neural network algorithm is trained using the fit_generator method from Keras. To improve the model's robustness and generalization, the training data undergoes data augmentation techniques, including rescaling, shear range, zoom range, and horizontal flipping. The model is trained over 10 epochs with a batch size of 10.
During the training process, the ModelCheckpoint callback saves the best model based on validation accuracy. Additionally, the entire trained model is saved at the end of training for future use.

## Testing
To evaluate the model's performance on your custom dataset, follow these steps:

Prepare your dataset and organize it into separate directories for training and testing.
Update the file paths in the provided testing script accordingly.
Run the testing script:

## License
This project is licensed under the [MIT License](LICENSE). Published as - Materials 2023, 16(15), 5426; https://doi.org/10.3390/ma16155426

