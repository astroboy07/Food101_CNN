from contextlib import suppress
import streamlit as st
import tensorflow as tf
import pandas as pd
from utility import load_prep, class_names

st.set_page_config(page_title="Food Recognition",
                    page_icon="🍕")

# prediction made by tensorflow model
@st.cache(suppress_st_warning=True)
def model_prediction(image, model):
    img = load_prep(image)
    img = tf.expand_dims(img, axis=0)
    pred_prob_image = model.predict(img)
    pred_class_image = class_names[pred_prob_image.argmax()]
    
    top_3_pred_prob_idx = (pred_prob_image.argsort())[0][-3:][::-1]
    top_3_pred_prob = [pred_prob_image[0][idx1] for idx1 in top_3_pred_prob_idx]
    top_3_pred_class = [class_names[idx2] for idx2 in top_3_pred_prob_idx]
    df = pd.DataFrame({"Top 3 predictions": top_3_pred_class,
                    "F1 scores": top_3_pred_prob})
    return pred_class_image, pred_prob_image, df



# Main body
st.title("Food Recognition 🍕🔍")
st.header("Recognize the food images! 🕵🏻‍♂️")

## upload food image
image_file = st.file_uploader(label="Choose a file",
                        type=['png', 'jpg'])


# tensorflow model for prediction
dirpath = "/Users/saifali/Downloads/Machine Learning Projects/Food101_TransferLearning/"
model = tf.keras.models.load_model(dirpath + "Saved_model/fine_tuned.hdf5")

# if file is not uploaded, stop the process
if not image_file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = image_file.read()
    st.image(image, use_column_width="auto")
    go = st.button("Go!", help="Press the button to recognize food.")

if go:
    pred_class_image, pred_prob_image, df = model_prediction(image, model)
    st.success(f"Predicted food: {pred_class_image} \n Prediction percentage = {pred_prob_image * 100:.1f}", icon="✅")
    