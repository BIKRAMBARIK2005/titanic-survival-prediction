import streamlit as st
import pandas as pd
import pickle
import time
import base64
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1 { font-weight: 700; }
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.card {
    padding: 1.5rem;
    border-radius: 12px;
    background-color: #1c1f26;
    margin-bottom: 1rem;
}
.success {
    background-color: #1e3d2f;
    padding: 1.5rem;
    border-radius: 12px;
    color: #00ff99;
    text-align:center;
    font-size:20px;
}
.fail {
    background-color: #3d1e1e;
    padding: 1.5rem;
    border-radius: 12px;
    color: #ff4b4b;
    text-align:center;
    font-size:20px;
}
.footer {
    text-align:center;
    padding:20px;
    font-size:14px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("models_with_pipeline/pipe.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚢 Titanic AI")
page = st.sidebar.radio("Navigation", ["Home", "Predict", "History", "About"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🚢 Titanic AI Survival Predictor")
    st.markdown("""
    ### Production-Level Machine Learning SaaS Application
    
    ✔ Real ML Model  
    ✔ Probability Prediction  
    ✔ Fast Performance  
    ✔ Modern UI  
    ✔ Prediction History  
    
    Built with Streamlit + Scikit-learn.
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "Predict":

    st.title("🎯 Predict Survival")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1,2,3])
        sex = st.selectbox("Sex", ["male","female"])
        age = st.slider("Age", 1, 80, 25)

    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0,10,0)
        parch = st.number_input("Parents/Children", 0,10,0)
        fare = st.number_input("Fare", 0.0,600.0,50.0)
        embarked = st.selectbox("Embarked", ["S","C","Q"])

    if st.button("🚀 Run Prediction"):

        input_df = pd.DataFrame({
            "Pclass":[pclass],
            "Sex":[sex],
            "Age":[age],
            "SibSp":[sibsp],
            "Parch":[parch],
            "Fare":[fare],
            "Embarked":[embarked]
        })

        with st.spinner("Running AI model..."):
            time.sleep(1)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.history.append({
            "Time": timestamp,
            "Class": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Survival Probability": f"{probability*100:.2f}%"
        })

        st.markdown("---")

        if prediction == 1:
            st.markdown(f"""
            <div class="success">
            🎉 Passenger Would SURVIVE <br>
            Probability: {probability*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="fail">
            ❌ Passenger Would NOT Survive <br>
            Probability: {probability*100:.2f}%
            </div>
            """, unsafe_allow_html=True)

# ---------------- HISTORY PAGE ----------------
elif page == "History":

    st.title("📊 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# ---------------- ABOUT PAGE ----------------
elif page == "About":

    st.title("ℹ About This SaaS")

    st.markdown("""
    This application uses a trained Machine Learning Pipeline model
    to predict Titanic survival probability.

    Built by **Bikram Barik** 🚀

    Tech Stack:
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Made with ❤️ by Bikram Barik | Production SaaS Version
</div>
""", unsafe_allow_html=True)
