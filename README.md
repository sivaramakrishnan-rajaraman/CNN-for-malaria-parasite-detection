# Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images

This repository contains the Jupyter Notebooks to train custom CNNs and extract features from the underlying data using pretrained models applied to the challenge of Malaria Cell classification. These codes are from the following publication: 

Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018) Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images. PeerJ6:e4568 https://doi.org/10.7717/peerj.4568

Malaria is a blood disease caused by the Plasmodium parasites transmitted through the bite of female Anopheles mosquito. Microscopists commonly examine thick and thin blood smears to diagnose disease and compute parasitemia. However, their accuracy depends on smear quality and expertise in classifying and counting parasitized and uninfected cells. Such an examination could be arduous for large-scale diagnoses resulting in poor quality. State-of-the-art image-analysis based computer-aided diagnosis methods using machine learning techniques, applied to microscopic images of the smears using hand-engineered features demand expertise in analyzing morphological, textural, and positional variations of the region of interest. In contrast, Convolutional Neural Networks, a class of deep learning models promise highly scalable and superior results with end-to-end feature extraction and classification. Automated malaria screening using deep learning techniques could, therefore, serve as an effective diagnostic aid. In this study, we evaluate the performance of pre-trained models as feature extractors toward classifying parasitized and uninfected cells to aid in improved disease screening. We experimentally determine the optimal model layers for feature extraction from the underlying data. Statistical validation of the results demonstrates the use of pre-trained convolutional neural networks as a promising tool for feature extraction for this purpose.

# Data Availability
The segmented cells from the thin blood smear slide images for the parasitized and uninfected classes are made available at https://ceb.nlm.nih.gov/repositories/malaria-datasets/. To reduce the burden for microscopists in resource-constrained regions and improve diagnostic accuracy, researchers at the Lister Hill National Center for Biomedical Communications, part of National Library of Medicine, have developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected and photographed at Chittagong Medical College Hospital, Bangladesh. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. The de-identified images and annotations are archived at NLM. We applied a level-set based algorithm to detect and segment the red blood cells. The dataset consists of 27,558 cell images with equal instances of parasitized and uninfected cells. Positive samples contained Plasmodium and negative samples contained no Plasmodium but other types of objects including staining artifacts/impurities. We evaluated the predictive models through five-fold cross-validation. Cross-validation has been performed at the patient level to ensure alleviating model biasing and generalization errors. The data appears along with the publication:

Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018) Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images. PeerJ6:e4568 https://doi.org/10.7717/peerj.4568

The publication is also included to this repository for the readers' convenience. The images were re-sampled to 100 x 100, 224 x 224, 227 x 227 and 299 x 299-pixel resolutions to suit the input requirements of customized and pre-trained CNNs and mean normalized to assist in faster convergence. 

# Prerequisites
Keras 2.2.0

Tensorflow-GPU 1.9.0

Scikit-Learn

OpenCV

Matplotlib

# Pre-trained CNN models
We evaluated the performance of pre-trained CNNs including AlexNet (winner of ILSVRC 2012), VGG-16 (winner of ILSVRC's localization task in 2014), Xception, ResNet-50 (winner of ILSVRC 2015) and DenseNet-121 (winner of the best paper award in CVPR 2017) toward extracting the features from the parasitized and uninfected cells. We instantiated the convolutional part of the pre-trained CNNs and trained a fully-connected model with dropout (dropout ratio of 0.5) on top of the extracted features. We also empirically determined the optimal layer for feature extraction to aid in improved classification. The custom and pre-trained models are optimized for hyper-parameters by a randomized grid search method. We evaluated the performance of the CNNs in terms of accuracy, AUC, sensitivity, specificity, F1-score, and MCC. The model architecture and weights for the pre-trained CNNs were downloaded from GitHub repositories (Chollet, 2017; Yu, 2016).
