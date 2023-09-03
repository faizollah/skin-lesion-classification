# Skin Lesion Classification
This project aims to classify skin lesions using various deep learning models. The dataset used is the HAM10000 dataset from Kaggle.

## Dataset
The [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) consists of dermatoscopic images of various skin lesions:

- akiec - Actinic keratoses
- bcc - Basal cell carcinoma
- bkl - Benign keratosis
- df - Dermatofibroma
- mel - Melanoma
- nv - Melanocytic nevi
- vasc - Vascular lesions
Each image is associated with a label from the above categories.

## Data Processing
1. The metadata is loaded and paths to the images are generated.
2. The dataset is split into training, validation, and test sets.
3. Images are preprocessed: loaded, resized, and normalized.
4. The preprocessed images and their labels are saved in .npy format for faster access in future runs.

## Models
Several models are explored for this classification task:

- Custom CNN Model: A custom convolutional neural network model with multiple convolutional layers, max-pooling, batch normalization, and dropout.
- Transfer Learning with Xception: The Xception model pre-trained on ImageNet is fine-tuned for the skin lesion classification task.
- Transfer Learning with EfficientNetB0: The EfficientNetB0 model pre-trained on ImageNet is used with data augmentation and class weighting to address class imbalance.
- Ensemble Learning with InceptionV3, ResNet152, and DenseNet201: Predictions from three different models are combined to make a final prediction.
- InceptionV3 with Data Augmentation: The InceptionV3 model is trained with augmented data to improve generalization.

## Evaluation
Model performance is evaluated using accuracy, classification reports, and confusion matrices.

