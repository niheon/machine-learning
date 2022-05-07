## Table of Contents

- [Chest X-Ray Medical Diagnosis with Deep Learning](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Diagnosis/Part%201%20-%20Chest%20X-Ray%20Medical%20Diagnosis%20with%20Deep%20Learning.ipynb)
- [Brain Tumor Auto-Segmentation for Magnetic Resonance](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Diagnosis/Part%202%20-%20Brain%20Tumor%20Auto-Segmentation%20for%20Magnetic%20Resonance%20Imaging%20(MRI).ipynb)
- [Evaluation of Diagnostic Models](https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Diagnosis/Part%203%20-%20Evaluation%20of%20Diagnostic%20Models.ipynb)

## Chest X-Ray Medical Diagnosis with Deep learning

<img src="https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Diagnosis/chestxray.png" alt="Chest X-ray Medical Diagnosis">

The goal is to diagnose diseases from chest X-rays using a deep learning model. This project employs a DenseNet-121 model that has been pre-trained to identify 14 labels, including Cardiomegaly, Mass, and Pneumothorax. GradCAM is used to show and highlight where the model is looking and which area of interest is used to create the forecast.

Chest x-ray pictures from the public [ChestX-ray8](https://arxiv.org/abs/1705.02315) collection are used in this project. There are 108,948 frontal-view X-ray pictures in this dataset, representing 32,717 different patients. Multiple text-mined labels identify 14 different disease states in each image in the data set. A subset of 1000 photos is used for this project.

## Brain Tumor Auto-Segmentation for Magnetic Resonance Imaging

<img src="https://github.com/rajeshai/machine-learning/blob/main/AI%20for%20Medicine/AI%20for%20Medical%20Diagnosis/braintumor.jpg" alt="Brain Tumor Segmentation">

The goal of the project is to build a neural network to automatically segment tumor regions in the brain, using MRI (Magnetic Resonance Imaging) scans. This project employs a 3D U-net model. This architecture will take advantage of the volumetric shape of MR images and is one of the best-performing models for this task. Feel free to familiarize yourself with the architecture by reading [this paper](https://arxiv.org/abs/1606.06650)

The dataset used in this project is from the [Decathlon 10 Challenge](https://decathlon-10.grand-challenge.org/). This data has been mostly pre-processed for the competition participants, however, in real practice, MRI data needs to be significantly pre-preprocessed before we can use it to train our models and have access to a total of 484 images for training.

## Evaluation of Diagnostic Models

The goal of the project is to evaluate the diagnostic models we built, more specifically working with the results of the X-ray classification model we built previously. Different metrics like True Positives, False Positives, True Negatives, False Negatives, Accuracy, Prevalence, etc., are discussed and evaluated.



