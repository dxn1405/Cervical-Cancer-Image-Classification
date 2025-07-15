import os
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

# Import your ViT model classes
from models.swin import SwinTransformer
from models.DeiT import DeiT
from models.crossViT import CrossViT

# ——— Streamlit page setup ———
st.set_page_config(page_title="Classify Image", layout="wide")
st.sidebar.title("Cervical Cancer Image Classification Using Vision Transformer")


st.title("Insert Your Cervical Cancer Image")
st.caption("Supported formats: .jpeg, .jpg, .png, .bmp")

# single-image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpeg","jpg","png","bmp"])

# model choice
model_name = st.radio(
    "Select Model",
    ["Swin Transformer", "DeiT", "CrossViT"],
    horizontal=True
)

# your five classes
class_names = ['Dyskeratotic','Koilocytotic','Metaplastic','Parabasal','Superficial']

# standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@st.cache_resource
def load_model(chosen_label: str):
    """Map the radio label → checkpoint name + model class, load and return."""
    ckpt_map = {
        "Swin Transformer": ("Swin",    SwinTransformer),
        "DeiT":              ("DeiT",    DeiT),
        "CrossViT":          ("CrossViT",CrossViT),
    }
    if chosen_label not in ckpt_map:
        raise ValueError(f"Unknown model label: {chosen_label}")

    ckpt_name, ModelClass = ckpt_map[chosen_label]
    path = os.path.join("models", "trained_models", f"{ckpt_name}_best.pth")


    model = ModelClass(num_classes=len(class_names))
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Run Classification"):
        with st.spinner("Classifying..."):
            # 1) make batch
            x = transform(img).unsqueeze(0)   # [1,3,224,224]

            # 2) load & forward
            model = load_model(model_name)
            out   = model(x)

            # 3) collapse spatial dims if present
            if out.dim() > 2:
                out = out.view(out.size(0), out.size(1), -1).mean(dim=2)

            # 4) softmax + detach
            probs = F.softmax(out, dim=1)[0].cpu().detach().numpy()
            pred  = class_names[probs.argmax()]

        # —— display results ——
        st.success("✅ Classification Complete")
        st.subheader(f"Predicted Class: **{pred}**")

        st.markdown("### Confidence Scores")
        for label, conf in zip(class_names, probs):
            st.write(f"- {label}: {conf:.4f}")

        st.markdown("### Visual Confidence Chart")
        n_labels = len(class_names)
        if len(probs) < n_labels:
            st.error(f"❌ Only got {len(probs)} scores for {n_labels} labels.")
            st.stop()

        plot_probs = probs[:n_labels]

        fig, ax = plt.subplots()
        ax.barh(class_names, plot_probs)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")
        ax.invert_yaxis()
        st.pyplot(fig)

else:
    st.info("👈 Please upload an image to start classification.")
