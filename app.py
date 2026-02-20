import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# Page config

st.set_page_config(
page_title="Logic Gate Classifier",
page_icon="🧠",
layout="centered"
)

# Load model and class names

@st.cache_resource
def load_model():
model = tf.keras.models.load_model("logic_gate_classifier.h5")
with open("class_names.json", "r") as f:
class_names = json.load(f)
return model, class_names

model, class_names = load_model()

IMG_SIZE = 64

# Gate descriptions

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

# Title

st.title("Logic Gate Symbol Classifier")
st.write("Upload a hand-drawn logic gate symbol to classify.")

# Upload image

uploaded_file = st.file_uploader(
"Upload Image",
type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

```
# Display image
image = Image.open(uploaded_file).convert("L")
st.image(image, caption="Uploaded Image", width=200)

# Preprocess
img = np.array(image)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)
confidence = prediction[0][class_index]

gate_name = class_names[class_index]

# Display result
st.success(f"Detected Gate: {gate_name}")
st.info(f"Confidence: {confidence*100:.2f}%")

# Educational feedback
st.subheader("Educational Feedback")
st.write(gate_info.get(gate_name, "No information available."))

# Confidence feedback
if confidence > 0.9:
    st.success("Excellent drawing! Very accurate.")
elif confidence > 0.7:
    st.warning("Good drawing, but could be improved.")
else:
    st.error("Low confidence. Try drawing more clearly.")
```
