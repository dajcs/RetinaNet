# RetinaNet
CVIA project competition

This package has been put together in order to execute the CVIA project and to Participate in the [Spark challenge](https://gitlab.uni.lu/spark-challenge/2022-utils).

Started with simple experiments presented in a [fast.ai course](https://forums.fast.ai/t/part-2-lesson-9-wiki/14028)

The results of the experiments can be seen in `fastai_experiments.ipynb`.  This experiment is based on ResNet34 and the last layer contains a detector for a single bbox (4 activations) and an image classification with cross-enthropy (11 activations for the 11 classes).

Note: this repository contains only a subset of the dataset.  The whole dataset is needed to reproduce the results in the notebook.  The best result what we could achieve on the [codalab competition](https://codalab.lisn.upsaclay.fr/) was about 0.42



Continued by examining the possiblities of a RetinaNet model, based on [this](https://debuggercafe.com/train-pytorch-retinanet-on-custom-dataset/) code.

The code has been modified to our needs (single bbox detection and additional insights about the data) and the image augmentations have been enhanced.  The best result achieved on the codalab competition is about 0.84

