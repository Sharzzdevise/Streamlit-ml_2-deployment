import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img

# Load the model
model = tf.keras.models.load_model("c:\\Users\\dimsh\\my_app\\fine_tuned_model")

def main():
    st.title("Breast Cancer Image Detection and Classification System")

    # Upload an image for classification
    uploaded_file = st.file_uploader("C:\\Users\\dimsh\\my_app\\Static\\services-body-mammography.png", type=["png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img_array = np.array(image.resize((224, 224)))
        img_array = img.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = img_array / 255.0  # Normalize pixel values if required

        # Perform prediction
        prediction = model.predict(processed_img)

        # Display prediction results
        st.subheader("Prediction Results:")
        if prediction[0][0] > 0.5:  # Assuming it's binary classification
            st.write("Predicted class: Class Normal")
        else:
            st.write("Predicted class: Class Cancerous")

if __name__ == '__main__':
    main()