import streamlit as st
import tensorflow as tf
import pandas as pd


# class names of 101 foods
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 
                'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake',
                'ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla',
                'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
                'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes',
                'deviled_eggs','donuts','dumplings','edamame','eggs_benedict',
                'escargots','falafel','filet_mignon','fish_and_chips','foie_gras',
                'french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice',
                'frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
                'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
                'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna',
                'lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons',
                'miso_soup','mussels','nachos','omelette','onion_rings',
                'oysters','pad_thai','paella','pancakes','panna_cotta',
                'peking_duck','pho','pizza','pork_chop','poutine',
                'prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake',
                'risotto','samosa','sashimi','scallops','seaweed_salad',
                'shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak',
                'strawberry_shortcake','sushi','tacos','takoyaki','tiramisu',
                'tuna_tartare','waffles']

st.set_page_config(page_title="Food Recognition",
                    page_icon="🍕")


# defining necessary functions

def load_prep(image, img_shape=224, scale=True):
    img = tf.io.decode_image(image, channels=3)
    img = tf.image.resize(img, size=([img_shape, img_shape]))
    if scale:
        return img/255.
    return img

@st.cache(suppress_st_warning=True)
def predicting(img, model):
    img = load_prep(img, scale=False)
    img = tf.cast(tf.expand_dims(img, axis=0), tf.int16)
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
st.header("Recognize food images! 🕵🏻‍♂️")


## upload food image
image_file = st.file_uploader(label="Choose a file",
                        type=['png', 'jpg'])


dirpath = "/Users/saifali/Downloads/Machine_Learning_Projects/Food101_TransferLearning/Extras/fine_tuned"
model = tf.keras.models.load_model(dirpath)


if not image_file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = image_file.read()
    st.image(image, use_column_width=True)
    go = st.button("Go!", help="Press the button to recognize food.")

if go:
    pred_class_image, pred_prob_image, df = predicting(image, model)
    st.success(f"Predicted food: {pred_class_image} | Prediction percentage = {pred_prob_image.max() * 100:.1f}", icon="✅")
    
    # st.write(alt.Chart(df).mark_bar().encode(
    #     x='F1 Scores',
    #     y=alt.X('Top 3 Predictions', sort=None),
    #     text='F1 Scores'
    # ).properties(width=600, height=400))
