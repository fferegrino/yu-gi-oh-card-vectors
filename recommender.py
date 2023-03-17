from tempfile import NamedTemporaryFile

import boto3

import streamlit as st
import json
from annoy import AnnoyIndex

st.set_page_config(layout="wide")


@st.cache_resource
def load_model():
    model_ann = AnnoyIndex(30, "angular")
    s3 = boto3.client("s3")
    bucket_name = "feregrino-metaflow-experiments"
    # Download the model to a temporary file
    with NamedTemporaryFile() as tmp:
        object_key = "yu-gi-oh/index.ann"
        s3.download_fileobj(bucket_name, object_key, tmp)
        model_ann.load(tmp.name)

    object_key = "yu-gi-oh/cards.json"
    with open("cards.json", "wb") as f:
        s3.download_fileobj(bucket_name, object_key, f)
    with open("cards.json") as f:
        id_to_card = json.load(f)
    id_to_card = {int(k): v for k, v in id_to_card.items()}
    return model_ann, id_to_card


model, id_to_card = load_model()

names = ["Select a card..."]
passcodes = []

for idx, card in id_to_card.items():
    names.append(card["name"])
    passcodes.append(int(card["passcode"]))

st.title("Yu-Gi-Oh! Card Recommender")

selected = st.selectbox("Select a card", [idx for idx, _ in enumerate(names)], format_func=lambda idx: names[idx])
if selected > 0:
    selected = selected - 1
    selected_card = id_to_card[selected]
    found_cards, distances = model.get_nns_by_item(selected, 6, search_k=-1, include_distances=True)
    st.subheader(f"Recommended cards for {selected_card['name']}")
    cols = st.columns(len(found_cards))
    for idx, (card_id, distance, col) in enumerate(zip(found_cards, distances, cols)):
        passcode = passcodes[card_id]
        if idx == 0:
            caption = f"**Selected card**"
        else:
            caption = f"Distance - **{distance:.2f}**"
        with col:
            st.image(f"https://images.ygoprodeck.com/images/cards/{passcode}.jpg")
            st.markdown(caption)
else:
    st.subheader("Select a card to see similar cards")
