import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------------------------------
# Page Config (Must be first Streamlit command)
# ---------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom Premium Styling
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    font-weight: 700;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.success {
    background-color: #1e3d2f;
    color: #00ff88;
}
.fail {
    background-color: #3d1e1e;
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Cached Model Loading (⚡ Speed Boost)
# ---------------------------------------------------
@st.cache_resource
def load_models():
    pipe = pickle.load(open("models_with_pipeline/pipe.pkl", "rb"))
    return pipe

pipe = load_models()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("⚙ Model Options")
st.sidebar.info("Pipeline Model is recommended for best performance.")

# ---------------------------------------------------
# Main Title
# ---------------------------------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Predict whether a passenger would survive the Titanic disaster.")

st.divider()

# ---------------------------------------------------
# Layout Columns
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

st.divider()

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button("🔍 Predict Survival"):

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]], dtype=object)
    prediction = pipe.predict(input_data)[0]

    if prediction == 1:
        st.markdown('<div class="result-box success">🎉 This passenger would SURVIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box fail">❌ This passenger would NOT survive</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ by Bikram Barik")
