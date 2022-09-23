import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

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

tab1, tab2, tab3 = st.tabs(["🔍 Food Recognition", "🛠 Details", "🎓 About me!"])

# in tab 1
# defining necessary functions

with tab1: 
    def load_prep(image, img_shape=224, scale=True):
        img = tf.io.decode_image(image, channels=3)
        img = tf.image.resize(img, size=([img_shape, img_shape]))
        if scale:
            return img/255.
        return img

    @st.cache(suppress_st_warning=True)
    def predicting(img, model):
        img = load_prep(img, scale=False)
        img = tf.expand_dims(img, axis=0)
        # img = tf.cast(tf.expand_dims(img, axis=0), tf.int16)
        pred_prob_image = model.predict(img)
        pred_class_image = class_names[pred_prob_image.argmax()]
        top_3_pred_prob_idx = (pred_prob_image.argsort())[0][-3:][::-1]
        top_3_pred_prob = [pred_prob_image[0][idx1] for idx1 in top_3_pred_prob_idx]
        top_3_pred_class = [class_names[idx2] for idx2 in top_3_pred_prob_idx]
        return pred_class_image, pred_prob_image, top_3_pred_class, top_3_pred_prob

    # Main body
    st.title("Food Recognition 🍕🔍")
    st.header("Recognize food images! 🕵🏻‍♂️")



    ## upload food image
    image_file = st.file_uploader(label="Choose a file",
                            type=['png', 'jpg'])

    # get the model
    dirpath_saved_model = "./Saved_model/fine_tuned.hdf5"
    model = tf.keras.models.load_model(dirpath_saved_model)

    # dirpath = "./Extras/fine_tuned"
    # model = tf.keras.models.load_model(dirpath)


    if not image_file:
        st.warning("Please upload an image")
        st.stop()

    else:
        image = image_file.read()
        st.image(image, use_column_width=True)
        go = st.button("Go!", help="Press the button to recognize food.")

    if go:
        pred_class_image, pred_prob_image, top_3_class, top_3_prob = predicting(image, model)
        st.success(f"Predicted food: {pred_class_image} | Prediction percentage = {pred_prob_image.max() * 100:.1f}%", icon="✅")
        fig, ax = plt.subplots()
        top_3_prob = [top_3_prob[i] * 100 for i in range(len(top_3_prob))] 
        ax.bar(top_3_class, top_3_prob)
        ax.set_title("Top 3 predictions")
        st.pyplot(fig)

with tab2:
    st.write("""
    An end-to-end machine learning model for recognizing food in your image. This model is trained on 
    [101 classes of food images](https://www.tensorflow.org/datasets/catalog/food101). It can detect food 
    items such as chicken curry, cheesecake, ramen, samosa, chocolate cake and many more! 

    The model is based fine-tuning of **EfficientNetB1** which was initially pre-trained on ImageNet. The overall 
    accuracy of the model is **80%**.

    For more details, please visit [**GitHub**](https://github.com/astroboy07/Food101_CNN).
    """)

with tab3:
    st.write("""
    I am a Ph.D. candidate in Department of Physics at University of Texas at Dallas. My research focuses on gravitational 
    lensing, gravitational waves (GWs), and black holes (BHs). In particular, I am working on the detectability of gravitational 
    lensing of GWs and quantify the precession signature on GWs emitted by BH system.
    """)