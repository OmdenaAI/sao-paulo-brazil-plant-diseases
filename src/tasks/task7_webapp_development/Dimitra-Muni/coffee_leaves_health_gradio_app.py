# -*- coding: utf-8 -*-
"""Coffee Leaves Health Gradio App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uGVlILd5kPhXleVDr9S9jO8RgvKy6VnR

## Task:7 Web App Development (Omdena São Paulo Chapter challenge)
  This colab notebook is created as the part of Omdena São Paulo Chapter challenge: **Classification of Plant Diseases in Brazilian Agriculture Using Computer Vision and Machine Learning**


  Collaborator: Dimitra Muni (muni.dimitra@gmail.com)
 - Objective:
  - To encourage the collaborators to develop their own version of Coffee Leaf Health prediction model.
  - Learn about Gradio Interface, and integrate it with the model.
"""

!pip install -q gradio

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import models,layers,utils
import gradio as gr

from google.colab import drive
drive.mount('/content/drive')

"Ensure that the Challenge short cut is added on your google drive in order to following command to work"
model=models.load_model('/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-4-Model(s) Building/Dimitra-Muni/model_CNN1_BRACOL.h5')

model.summary()

class_names=['healthy', 'miner', 'rust', 'phoma', 'cercospora']

#plt.imshow(img)
def predict(input_image):
    image=input_image.reshape(-1, 256, 256, 3)
    predictions=model.predict(image)[0].flatten()
    #class_confidence={class_names[i]:np.single(predictions[i]) for i in range(len(class_names)) }
    class_confidence=dict(zip(class_names,np.single(predictions) ))
    #return pd.DataFrame(class_confidence,columns=class_names)
    return class_confidence

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(shape=(256, 256)),
                    outputs=gr.Text(placeholder='Predicted Class Probabilities'),
                    title='Coffee Disease Classifier',
                    description='Disease Classification for the Coffee Leaves using CNN, trained using BRACOL dataset',
                    allow_flagging='never',
                    examples=['/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-7-Web App Development/Dimitra Muni/examples/cercospora.jpg',
                              '/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-7-Web App Development/Dimitra Muni/examples/healthy.jpg',
                              '/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-7-Web App Development/Dimitra Muni/examples/miner.jpg',
                              '/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-7-Web App Development/Dimitra Muni/examples/phoma.jpg',
                              '/content/drive/MyDrive/SaoPauloChapter_Plants-Disease_2023-Aug-02/Task-7-Web App Development/Dimitra Muni/examples/rust.jpg',
                              ])


if __name__ == "__main__":
    demo.launch(share=True)

"""# References

- 1  [Gradio Crash Course - Fastest way to build & share Machine Learning apps](https://youtu.be/eE7CamOE-PA?si=6Ydh_az-Ea9xuMk-)
- 2 [Image Classification in TensorFlow and Keras](https://www.gradio.app/guides/image-classification-in-tensorflow)

"""