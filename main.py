import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import PIL

st.header('Age Variation Detector', divider='rainbow')

with st.form("my_form"):

    st.write("Fill the below form.")

    st.write("Note: Upload the cropped image in which C2, C3 and C4 is visible.")
    uploaded_files = st.file_uploader("Select an image", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write(":blue[**File Uploaded**]", uploaded_file.name)
        st.image(bytes_data)

    st.divider()
    
    option = st.selectbox(
        "Type of patient",
        ("Cleft patient", "Non - cleft patient"),
        index=None
        )
    
    st.divider()

    age = st.text_input('Enter the chronological age of the patient', None)

    
    submitted = st.form_submit_button("Submit")

    ##########################################################

    if submitted:
        model = tf.keras.models.load_model('keras_model.h5', compile=False)
        class_names = open('labels.txt', 'r').readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

        new_size=(224, 224)
        img_resized = image.resize(new_size, PIL.Image.Resampling.LANCZOS)

    #   turn the image into a numpy array
        image_array = np.asarray(img_resized)

    #   Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    #   Load the image into the array
        data[0] = normalized_image_array

    #   Predicts the model
        prediction = model.predict(data)

        index = np.argmax(prediction)

        class_name = class_names[index]

        confidence_score = prediction[0][index]


        # ###########

        model2 = tf.keras.models.load_model('keras_model2.h5', compile=False)
        class_names2 = open('labels2.txt', 'r').readlines()

    #   Predicts the model
        prediction2 = model2.predict(data)

        index2 = np.argmax(prediction2)

        class_name2 = class_names2[index2]

        confidence_score = prediction2[0][index2]

        st.write(':blue[**Type of patient:**]', option)
        st.write(':blue[**Chronological age of the patient:**]', f"{age} years")
        st.write(':blue[**Skeletal age of the patient:**]', class_name[2:])
        st.write(':blue[**Cervical vertebral anomalies:**]', class_name2[2:])







