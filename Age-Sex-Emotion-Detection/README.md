# Sex, Age and Emotion detection using deep learning

## Introduction

This project aims to classify the Age, Sex and Emotion on a person's face into one of seven categories, using deep 
convolutional neural networks. 

There is two models used the first one is trained on the FER-2013 dataset which was published on International 
Conference on Machine Learning (ICML). And the second one is based on FairFace dataset which is race balanced. It
contains 108,501 images from 7 different race groups

## How To Run:
* 1- Install requirements.
* 2- change the path to the (state_dict_model.pt) path.
* 3- Run using command : 
```bash
cd src
python emotions.py --mode display
```