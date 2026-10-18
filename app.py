import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

@st.cache_resource
def load_artifacts():
    with open("titanic_model_bundle.pkl", "rb") as file:
        return pickle.load(file)

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    pclass_encoder = artifacts["pclass_encoder"]
    gender_encoder = artifacts["gender_encoder"]
    sibling_encoder = artifacts["sibling_encoder"]
    embarked_encoder = artifacts["embarked_encoder"]
    survived_encoder = artifacts["survived_encoder"]
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("Enter passenger details to predict the model's result.")

with st.form("passenger_form"):
    pclass = st.selectbox(
        "Passenger Class", ["First", "Second", "Third"]
    )
    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )
    sibling = st.selectbox(
        "Sibling Count", ["Zero", "One", "Two", "Three"]
    )
    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton", "Cherbourg", "Queenstown"]
    )

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    try:
        input_data = pd.DataFrame([{
            "PClass": pclass_encoder.transform([pclass])[0],
            "Gender": gender_encoder.transform([gender])[0],
            "Sibling": sibling_encoder.transform([sibling])[0],
            "Embarked": embarked_encoder.transform([embarked])[0]
        }])

        prediction = model.predict(input_data)[0]
        result = survived_encoder.inverse_transform([int(prediction)])[0]

        if result == "Yes":
            st.success("Prediction: Passenger is predicted to survive.")
        else:
            st.warning("Prediction: Passenger is predicted not to survive.")

        st.caption(
            "Educational demonstration only. This is a model prediction, "
            "not a guarantee of what would have happened."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
