import streamlit as st
from PIL import Image
from helping_functions import load_css

st.set_page_config(layout='wide')
load_css()

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

IMAGE_SIZE = (800, 800)
st.header("Model Details:")

col1, col2 = st.columns([2,1])
st.subheader("Siamese network (MobileNetV2 architecture)")
image6 = Image.open('assets/Images/image6.png').resize(IMAGE_SIZE)
st.image(image6)
st.subheader("Working")
st.write("The Siamese Network architecture is used for the coffee disease classification using images. \
The main goal for using this network is to build an embedding space where the different classes are separated into distinct clusters. \
The siamese network architecture is used to generate feature embeddings such that intra-class image features are close to each other in the embedding space, \
    whereas the inter-class image features are further apart from each other. \
    The motivation behind this setup is to allow diverse input images that are semantically similar to be close to each other. \
    In the coffee dataset, we have single leaf images indoors on a white background under different lighting conditions, \
    as well as images outdoors in natural light on the plant. \
    Siamese networks capture this diversity while still keeping the features close to each other in the embedding space. \
    To measure the distance or closeness in embedding space, we use similarity metrics like cosine similarity, euclidean distance, and dot product. \
    For the coffee disease classification problem the euclidean distance measures the closeness between image pairs, which in technical terms is also called the similarity metric. \
    The advantage of using this method is 2-fold: \
    One can use a simple K-nearest neighbour approach to classify the input image. \
    The setup can visually explain why the model chose the particular class. \
    For classification using K-NN, the value of K is chosen to be 3. \
    ")

st.subheader("Fine-tuned ResNet 50")
st.subheader("Working")
st.write("Built on a Sequential model in Keras, initially incorporating a pre-trained ResNet model. \
Includes MaxPooling2D for downsampling and a Flatten layer for reshaping. \
Consists of three Dense layers with dropout layers in between for regularization; the neuron configurations are\
2048, 4096, and 2048 with ReLU activation.\
The output layer has 4 units with softmax activation, targeting 4 classes ['Healthy', 'Miner', 'Phoma', 'Rust'].\
Trained on the Cafe Arabic dataset with 3900 images over 10 epochs; batch size set at 128 for training and 512 for\
testing.\
Early stopping mechanism implemented to prevent overfitting.\
    ")

st.subheader("Customised CNN")
# image6 = Image.open('assets/Images/image6.png').resize(IMAGE_SIZE)
# st.image(image6)
st.subheader("Working")
st.write("Detailed description of the Customised CNN model Architecture:\
1. Rescaling Layer (Input Layer):\n\
This is the input layer, and it's not a part of the neural network itself. \
It simply scales the pixel values of the input images to a specified range.\
In this case, it scales the pixel values of the input images from [0, 255] to [0, 1], which is common for neural network input.\
\n\
2. Convolutional Layers:\n\
Each convolution has a ReLU activation function, which adds non-linearity to the model.\
The neural network architecture includes a series of convolutional layers with progressively increasing complexity. \
The first convolutional layer employs 32 filters of a 3x3 kernel size, resulting in 32 feature maps. \
These initial feature maps capture fundamental low-level image features like edges and basic textures, constituting a total of 896 parameters.\
Subsequently, the second convolutional layer applies 64 filters to the feature maps derived from the previous layer, generating an output shape of (32, 128, 128, 64) and totaling 18,496 parameters. \
This layer delves deeper to extract more intricate features.\
Moving forward, the third convolutional layer utilises 128 filters to capture higher-level image features, leading to an output shape of (32, 64, 64, 128) and a parameter count of 73,856.\
The fourth convolutional layer employs 192 filters for feature extraction, resulting in an output shape of (32, 32, 32, 192) and 221,376 parameters.\
The subsequent layers continue to increase in complexity: the fifth layer utilises 256 filters with an output shape of (32, 16, 16, 256) and 442,624 parameters, while the sixth layer employs 320 filters, \
resulting in an output shape of (32, 8, 8, 320) and 737,600 parameters.\
The seventh convolutional layer incorporates 384 filters, producing an output shape of (32, 4, 4, 384) and 1,106,304 parameters. \
Finally, the eighth layer employs 448 filters, yielding an output shape of (32, 2, 2, 448) and 1,548,736 parameters. \
These layers progressively capture and extract increasingly complex features from the input data, which is essential for the neural network's overall performance.\
\n\
3. Max-Pooling Layers\n\
In this neural network architecture, a series of eight Max-Pooling layers progressively reduce the spatial dimensions of the feature maps. \
Max-Pooling Layer 1, the initial layer, employs MaxPooling2D and yields an output shape of (32, 128, 128, 32), effectively reducing spatial dimensions while retaining essential information to alleviate computational demands. \
Subsequent Max-Pooling layers follow a similar pattern, with Layer 2 producing an output shape of (32, 64, 64, 64), and each subsequent layer further reducing spatial dimensions. \
Layer 8 serves as the final stage, compressing the feature maps to a 1x1 spatial dimension, resulting in an output shape of (32, 1, 1, 448). \
These Max-Pooling layers collectively contribute to feature reduction, a critical process in convolutional neural networks for feature extraction and dimensionality reduction.\
\n\
4. Flattening Layer\n\
The Flatten Layer takes the 1x1x448 feature maps and converts them into a 1D vector with an output shape of (32, 448), preparing them for input into the fully connected layers. \
5. Dense Layer 1\n\
Next, Dense Layer 1, with an output shape of (32, 64) and 28,736 parameters. This layer has 64 nodes followed by ReLU activation function. further processes these features.\
It's a fully connected layer with 64 neurons, contributing to feature refinement.\
\n\
6. Batch Normalisation\n\
 The Batch Normalisation Layer comes after, maintaining the output shape at (32, 64). \
It serves to normalise the activations from the previous layer, enhancing convergence speed and overall training stability. \
\n\
7. Dropout Layer\n\
The Dropout Layer, still with an output shape of (32, 64)Dropout is a regularisation technique that randomly drops a fraction of neurons during training with the probability 0.5, guarding against overfitting. \
 \n\
7. Dense Layer 2 \n\
Dense Layer 2, the output layer, has an output shape of (32, 5) with 325 parameters. It consists of 5 neurons, one for each class in the classification task, \
producing the final predictions. This layer uses Soft-Max activation to compute the probability for each of the five classes. \
This model consists of a series of convolutional and max-pooling layers for feature extraction, followed by fully connected layers for classification. \
Batch normalisation and dropout layers are used for regularisation and stability during training. \
")

# st.write("hello world")
