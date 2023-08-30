# cnn_image_classifier
CNN from scratch for image classification.

### METRICS

Accuracy: 94.72%

Confusion Matrix:
[[500   0   6   1   7]
 [  0 501   0   0   0]
 [ 28   0 446  15   2]
 [ 11   0  36 406  28]
 [  1   0   0   0 512]]
F1 Score: 0.9452815737800676
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.97      0.95       514
           1       1.00      1.00      1.00       501
           2       0.91      0.91      0.91       491
           3       0.96      0.84      0.90       481
           4       0.93      1.00      0.96       513

    accuracy                           0.95      2500
   macro avg       0.95      0.94      0.94      2500
weighted avg       0.95      0.95      0.95      2500

### Libraries:
The code uses popular libraries like PyTorch, PIL, scikit-learn, matplotlib, and NumPy.
Data Preparation:
Data Augmentation and Transformation: Transformations like RandomResizedCrop, RandomHorizontalFlip, and normalization are defined to preprocess the images.

Load Data: A custom dataset class CoffeeLeafDataset is created to load images from a given folder and label them according to their sub-directory name. This custom dataset also filters out corrupted images.

Data Splitting: The dataset is split into training, validation, and test sets.

Data Loaders: PyTorch's DataLoader is used to create data loaders for training, validation, and test sets.

### Model Architecture:
The model (CoffeeLeafClassifier) consists of:

Three convolutional layers followed by a ReLU activation and max-pooling.
Three fully connected layers to reduce the dimensions and output class scores.
Training:
Device: Checks if a GPU is available and sets it as the device for computation.

Hyperparameters: Defines batch size, learning rate, and number of epochs.

Optimization and Loss: Uses Adam optimizer and Cross-Entropy Loss.

Training Loop: Trains the model and validates it after each epoch. Implements early stopping based on validation loss.

### Metrics Evaluation:
Accuracy: Calculates the accuracy of the model on the test set.

F1 Score and Confusion Matrix: Uses scikit-learn to calculate the F1 Score and generate a confusion matrix.

Classification Report: Prints out a detailed classification report including precision, recall, and F1 score for each class.

Model Saving: The trained model is saved as a .pth file for future use.

### Inference:
Class Mappings: Creates a mapping from class index to class names.

Single Image Prediction: Loads a sample image and runs it through the model to predict its class.

Batch Prediction and Visualization: Takes a batch of 15 images from the test set, predicts their classes, and visualizes them along with their actual and predicted labels using matplotlib.
