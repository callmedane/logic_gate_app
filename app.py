import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

st.set_page_config(
    page_title="Logic Gate Classifier",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("logic_gate_classifier.h5")
    
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    
    return model, class_names

model, class_names = load_model()

IMG_SIZE = 64

gate_info = {
    "AND": "Outputs TRUE only if both inputs are TRUE.",
    "OR": "Outputs TRUE if at least one input is TRUE.",
    "NOT": "Outputs the opposite of the input.",
    "NAND": "Outputs FALSE only if both inputs are TRUE.",
    "NOR": "Outputs TRUE only if both inputs are FALSE.",
    "XOR": "Outputs TRUE if inputs are different.",
    "XNOR": "Outputs TRUE if inputs are the same.",
    "BUFFER": "Outputs the same as input."
}

st.title("Logic Gate Symbol Classifier")

uploaded_file = st.file_uploader("Upload Logic Gate Image")

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, width=200)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    gate = class_names[index]

    st.success(f"Detected Gate: {gate}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    st.write("Explanation:")
    st.write(gate_info.get(gate, "No description"))
