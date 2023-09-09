import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import tensorflow as tf

# Sequential model (Conv2D + MaxPool)
MODEL1 = tf.keras.models.load_model("Omdena_model1.h5", compile=False)
# Mobilenet-v2 
MODEL4 = tf.keras.models.load_model("Omdena_model4.h5", compile=False)

CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

#Function to get prediction array for a model (used in ensembling)
def get_all_predictions(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    return predictions[0]

#Function to get final predictions
def predict():
    upload = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])
    
    if upload is not None:
        image = Image.open(upload)
        #image = image.crop((left, top, right, bottom))
        st.image(image)
        newsize = (256, 256)
        image = image.resize(newsize)
        image = np.asarray(image)
        img_batch = np.expand_dims(image, 0)

        #Get model predictions
        predictions1 = MODEL1.predict(img_batch)
        predictions4 = MODEL4.predict(img_batch)
        
        #Get model predictions for ensemble output
        all_predictions1 = get_all_predictions(MODEL1, image)
        all_predictions4 = get_all_predictions(MODEL4, image)
        all_predictions_ensemble = (all_predictions1 + all_predictions4)/2

        #Get final prediction
        predicted_class1 = CLASS_NAMES[np.argmax(predictions1[0])]
        confidence1 = np.max(predictions1[0])

        predicted_class4 = CLASS_NAMES[np.argmax(predictions4[0])]
        confidence4 = np.max(predictions4[0])

        #Get final prediction for ensemble
        predicted_class_ensemble = CLASS_NAMES[np.argmax(all_predictions_ensemble[0])]
        confidence_ensemble = None
        
        return {"class1": predicted_class1, "confidence1": float(confidence1), "class4": predicted_class4, "confidence4": float(confidence4), "class_ensemble": predicted_class_ensemble, "confidence_ensemble": confidence_ensemble}
    else:
        return {"class1": "No Image", "confidence1": 0, "class4": "No Image", "confidence4": "No Image", "class_ensemble": "No Image", "confidence_ensemble": "No Image"}

predicted_output = predict()
st.write("Prediction from baseline CNN model (183877 parameters): ", predicted_output['class1'])
st.write("Prediction from Mobilenet-v2 (2667589 parameters): ", predicted_output['class4'])
st.write("Prediction from Ensemble of baseline and mobilenet-v2 : ", predicted_output['class_ensemble'])


