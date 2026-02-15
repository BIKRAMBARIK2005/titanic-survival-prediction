import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ------------------------
# Sidebar Model Selection
# ------------------------
st.sidebar.title("Model Selection")
model_type = st.sidebar.radio(
    "Choose Model:",
    ("Pipeline Model (Recommended)", "Without Pipeline Model")
)

# ------------------------
# Title
# ------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would survive the Titanic disaster.")

# ------------------------
# User Inputs
# ------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Number of Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# ------------------------
# Prediction Button
# ------------------------
if st.button("Predict Survival"):

    if model_type == "Pipeline Model (Recommended)":
        pipe = pickle.load(open('models_with_pipeline/pipe.pkl', 'rb'))
        data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]], dtype=object)
        prediction = pipe.predict(data)[0]

    else:
        ohe_sex = pickle.load(open('models_without_pipeline/ohe_sex.pkl', 'rb'))
        ohe_embarked = pickle.load(open('models_without_pipeline/ohe_embarked.pkl', 'rb'))
        clf = pickle.load(open('models_without_pipeline/clf.pkl', 'rb'))

        sex_encoded = ohe_sex.transform([[sex]]).toarray()
        embarked_encoded = ohe_embarked.transform([[embarked]]).toarray()

        final_input = np.concatenate(
            ([pclass, age, sibsp, parch, fare],
             sex_encoded[0],
             embarked_encoded[0])
        ).reshape(1, -1)

        prediction = clf.predict(final_input)[0]

    # ------------------------
    # Result Display
    # ------------------------
    if prediction == 1:
        st.success("🎉 The passenger would Survive!")
    else:
        st.error("💀 The passenger would NOT Survive.")
