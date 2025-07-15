import streamlit as st

st.set_page_config(page_title="About", layout="centered")

# Add this at the top of each page file
st.sidebar.title("Cervical Cancer Image Classification Using Vision Transformer")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/about.py", label="About")
st.sidebar.page_link("pages/UserGuideline.py", label="User Guideline")


st.title("What this project is about")

st.markdown("""
Cervical cancer is a major health issue, and early detection is key to saving lives.
Traditional diagnostic methods can be slow and prone to errors, which is where our solution comes in.

Our Cervical Cancer Image Classification System uses AI Models (DeiT, Swin Transformer, and CrossViT)
to automatically classify cervical cancer images. The system is fast, accurate, and easy to use,
helping medical professionals make quicker and more reliable diagnoses.

With a simple web interface, users can upload images and get instant results,
including the type of cancer cells and confidence levels.
""")
