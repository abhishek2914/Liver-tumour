import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array

st.write("""
            # Liver Cancer Detection
        """
         )

model_path = "/Users/abhishek/Desktop/liver_cancer_vgg19/tumour_vgg19.h5"

upload_file = st.sidebar.file_uploader("Upload cell images", type="jpg")

Generate_pred = st.sidebar.button("Predict")

model = tf.keras.models.load_model(model_path)

def import_n_pred(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis, ...]
    input_arr = img_to_array(img) / 255
    input_arr = np.expand_dims(input_arr, axis=0)
    pred = model.predict(input_arr)[0][0]
    return pred

if Generate_pred:
    if upload_file is not None:  # Check if file is uploaded
        image = Image.open(upload_file)
        with st.expander('Cell Image', expanded=True):
            st.image(image, use_column_width=True)
        print("Input image shape:", image.size)
        pred = import_n_pred(image, model)
        print("Prediction probabilities:", pred)
        labels = ["POSITIVE", "NEGATIVE"]
        if pred<0.5:
            print(st.title("Result: {}".format(labels[0])))
        else:
            print(st.title("Result: {}".format(labels[1])))

        result_index = np.argmax(pred)
    else:
        st.error("Please upload a file")

