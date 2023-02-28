# Contents of ~/my_app/pages/page_2.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model

# Load Model
model = load_model("weather_classification/weatherPrediction.h5")


def img_preprocessing(upload):
    img = Image.open(upload)
    image = tf.image.resize(img, (256, 256))
    image = np.expand_dims(image/255, 0)
    return image

def predict(image):
    st.image(image)
    img_fix = img_preprocessing(image)
    pred_weather = model.predict(img_fix).round()
    if pred_weather[0][0] == 1:
        weather = 'Cloudy'
    elif pred_weather[0][1] == 1:
        weather = 'Raining'
    elif pred_weather[0][2] == 1:
        weather = "Sunny"
    else:
        weather = "We're sorry, seems there are an error in our system"
    return weather
    

st.markdown("# Weather Detection ❄️")
st.caption("Let's play a game with today's weather :sunny: ")
st.write("This....")

img_type = ['jpg', 'jpeg', 'png']
uploaded_img = st.sidebar.file_uploader(
    label="Upload your image", type=img_type)

col1, col2 = st.columns(2)
with col1:
    st.write("Image Preview")
    if uploaded_img is not None:
       weather =  predict(uploaded_img)
    else:
        weather = predict("images/sunny.jpg")
with col2:
    st.write("What is the answer?")
    st.write("It is {}".format(weather))



