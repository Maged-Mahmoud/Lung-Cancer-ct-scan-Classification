# %% [code] {"execution":{"iopub.status.busy":"2022-06-30T18:06:15.798270Z","iopub.execute_input":"2022-06-30T18:06:15.798745Z","iopub.status.idle":"2022-06-30T18:06:15.806452Z","shell.execute_reply.started":"2022-06-30T18:06:15.798701Z","shell.execute_reply":"2022-06-30T18:06:15.805019Z"}}
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.applications import resnet, vgg16 , vgg19, densenet, efficientnet, mobilenet_v2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
import os
#import cv2
import streamlit as st

# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T14:32:18.86496Z","iopub.execute_input":"2022-06-29T14:32:18.865601Z","iopub.status.idle":"2022-06-29T14:32:33.756091Z","shell.execute_reply.started":"2022-06-29T14:32:18.865566Z","shell.execute_reply":"2022-06-29T14:32:33.754962Z"}}
ResNet_Path = './chest_CT_SCAN-ResNet50.hdf5'
DenseNet_Path = './chest_CT_SCAN-DenseNet201.hdf5'

ResNet_model = tf.keras.models.load_model(ResNet_Path)
DenseNet_model = tf.keras.models.load_model(DenseNet_Path)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-28T19:10:11.648162Z","iopub.execute_input":"2022-06-28T19:10:11.648628Z","iopub.status.idle":"2022-06-28T19:10:11.674564Z","shell.execute_reply.started":"2022-06-28T19:10:11.648537Z","shell.execute_reply":"2022-06-28T19:10:11.673493Z"}}
classes = {0:'adenocarcinoma',1:'large.cell.carcinoma',2:'normal',3:'squamous.cell.carcinoma'}

# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T14:45:37.097095Z","iopub.execute_input":"2022-06-29T14:45:37.097535Z","iopub.status.idle":"2022-06-29T14:45:37.119596Z","shell.execute_reply.started":"2022-06-29T14:45:37.097502Z","shell.execute_reply":"2022-06-29T14:45:37.118404Z"}}
# Predict Function
def predict(image):
    my_image = img_to_array(image)
    my_image = image.reshape((1, 460, 460,3))
    if np.max(ResNet_model.predict(my_image)) >=  np.max(DenseNet_model.predict(my_image)):
        y_pred = np.argmax(ResNet_model.predict(my_image))
    else:
        my_image = densenet.preprocess_input(my_image)
        y_pred = np.argmax(DenseNet_model.predict(my_image))
    
    return classes[y_pred]

# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T14:45:58.84557Z","iopub.execute_input":"2022-06-29T14:45:58.84607Z","iopub.status.idle":"2022-06-29T14:45:58.85054Z","shell.execute_reply.started":"2022-06-29T14:45:58.846006Z","shell.execute_reply":"2022-06-29T14:45:58.849725Z"}}
# Designing the interface
st.title("Lung Cancer CT-SCAN Classification App")
# For newline
st.write('\n')

img = Image.open('images/image.png')
show = st.image(img, use_column_width=True)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    show.image(image, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    image = tf.image.resize(image,(460,460))
    
# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            
            prediction = predict(image)
            time.sleep(2)
            st.success('Done!')
            
        st.sidebar.header("Algorithm Predicts: ")
                
        # Classify output 
        st.sidebar.write(f"It's an/a {str.upper(prediction)}.", '\n' )
                                                     
