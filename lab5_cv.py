import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# -------------------------------
# Step 1: Page configuration
# -------------------------------
st.set_page_config(
    page_title="Image Classification with ResNet18",
    layout="centered"
)

st.title("🖼️ Image Classification (ResNet18)")
st.write("This app uses a pre-trained ResNet18 model to classify images.")

# -------------------------------
# Step 2 & 3: Load model (CPU only)
# -------------------------------
device = torch.device("cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# -------------------------------
# Step 4 & 5: Image preprocessing
# -------------------------------
weights = models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

categories = weights.meta["categories"]

# -------------------------------
# Step 6: Image uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Step 7: Convert image to tensor
    # -------------------------------
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # -------------------------------
    # Step 8: Softmax + Top-5 results
    # -------------------------------
    probabilities = F.softmax(outputs[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = {
        "Class": [categories[i] for i in top5_catid],
        "Probability": [float(p) for p in top5_prob]
    }

    df = pd.DataFrame(results)

    st.subheader("🔍 Top-5 Predictions")
    st.table(df)

    # -------------------------------
    # Step 9: Bar chart visualization
    # -------------------------------
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(df.set_index("Class"))

# -------------------------------
# Step 10: Explanation
# -------------------------------
st.markdown("""
### 🔄 Classification Process
1. Image is uploaded by the user  
2. Image is resized and normalized  
3. Pre-trained ResNet18 extracts features  
4. Softmax converts outputs to probabilities  
5. Top-5 predictions are displayed  

This model runs fully on CPU and does not require training.
""")
