import streamlit as st

st.set_page_config(page_title="User Guideline", layout="centered")

# Add this at the top of each page file
st.sidebar.title("Cervical Cancer Image Classification Using Vision Transformer")


st.title("User Guideline")

st.markdown("""
### Upload Image
- Navigate to the homepage, click the "Classify" button.
- Upload cervical cancer cell image (PNG, JPG, JPEG only).

### Select AI Model
- Choose one from Swin Transformer, DeiT, or CrossViT.

### View Results
- Once classified, view the predicted class and model confidence scores.
- For example: *Dyskeratotic cell* with 93% confidence.
""")
