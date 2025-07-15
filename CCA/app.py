import streamlit as st
import base64
from download_models import download_all_models
download_all_models()

st.set_page_config(page_title="Cervical Cancer Classifier", layout="wide")

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_bg("assets/background.jpg")  # Replace with your actual image path

# Sidebar setup
st.sidebar.title("Cervical Cancer Image Classification Using Vision Transformer")
st.sidebar.page_link("pages/About.py", label="About")
st.sidebar.page_link("pages/UserGuideline.py", label="User Guideline")

# Hero section
st.markdown("""
    <h1 style='font-size:72px; color:white; font-weight:bold;'>A CLASSIFICATION SYSTEM</h1>
    <h3 style='font-size:36px; color:white;'>to detect types of cervical cancer cells</h3>
""", unsafe_allow_html=True)

if st.button("Classify"):
    st.switch_page("pages/Classify.py")
