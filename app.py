
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# --------------------------------------------
# App setup
# --------------------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #f4faf4 0%, #ffffff 55%, #eef7ed 100%);
}
[data-testid="stSidebar"] {
    background-color: #e8f3e8;
}
.hero {
    padding: 2rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #143d2b, #28784b);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { color: white; font-size: 2.8rem; }
.hero p { color: #e4f4e8; font-size: 1.1rem; }
.small-note { color: #53675a; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
# Model loading
# --------------------------------------------
@st.cache_resource
def load_classifier(model_file, model_type):
    checkpoint_path = ROOT / "models" / model_file

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("normalization_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization_std", [0.229, 0.224, 0.225])

    if model_type != "mobilenet_v2":
        raise ValueError(f"Unsupported architecture: {model_type}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, class_names, preprocessing, checkpoint


def pretty_leaf_name(name):
    if "___" not in name:
        return name.replace("_", " ")

    crop, disease = name.split("___", 1)
    crop = crop.replace("_(including_sour)", " (including sour)")
    crop = crop.replace("_(maize)", " (maize)")
    crop = crop.replace("Pepper,_bell", "Bell pepper")
    crop = crop.replace(",", "")
    crop = crop.replace("_", " ")

    disease = disease.replace("Haunglongbing", "Huanglongbing")
    disease = disease.replace("_", " ").strip()

    return f"{crop} | {disease}"


def predict(image, model_file, model_type):
    model, class_names, preprocessing, checkpoint = load_classifier(
        model_file, model_type
    )

    tensor = preprocessing(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0].cpu()

    indices = torch.argsort(scores, descending=True).tolist()

    return [
        (class_names[i], float(scores[i]))
        for i in indices
    ], checkpoint


def show_prediction(image, model_file, model_type, title, formatter):
    st.subheader(title)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        try:
            results, checkpoint = predict(image, model_file, model_type)
        except Exception as exc:
            st.error(f"Could not run the model: {exc}")
            st.stop()

    label, score = results[0]
    st.markdown("### Prediction")
    st.success(formatter(label))
    st.metric("Top model score", f"{score * 100:.2f}%")

    st.caption(
        "This score is the model's softmax output. It is not a calibrated "
        "probability or a guarantee of correctness."
    )

    st.markdown("### Top predictions")
    for class_name, value in results[:3]:
        st.write(f"**{formatter(class_name)}**")
        st.progress(value)
        st.caption(f"{value * 100:.2f}%")

    if model_type == "mobilenet_v2":
        st.caption(
            f"Model checkpoint: {checkpoint.get('model_name', model_type)}"
        )


# --------------------------------------------
# Navigation
# --------------------------------------------
st.sidebar.title("🌿 AgriVision AI")
st.sidebar.caption("AI-assisted plant and fruit analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Leaf Disease Detector",
        "Fruit Quality Detector",
        "Model Performance",
        "About & Limitations",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Compute device: {DEVICE.type.upper()}")


# --------------------------------------------
# Home
# --------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🌿 AgriVision AI</h1>
        <p>Explore plant leaf disease classification and fruit quality analysis
        with deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Upload a suitable image to receive a model prediction. "
        "This is a research prototype, not a replacement for expert "
        "agricultural assessment."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌱 Leaf disease detection")
        st.write(
            "Classifies supported plant leaves into 38 PlantVillage classes, "
            "including healthy and diseased categories."
        )
        if st.button("Open leaf detector", use_container_width=True):
            st.session_state["next_page"] = "Leaf Disease Detector"
            st.rerun()

    with col2:
        st.markdown("### 🍎 Fruit quality detection")
        st.write(
            "Classifies fruit images into Good, Bad, or Mixed quality labels "
            "from the FruitNet training dataset."
        )
        if st.button("Open fruit detector", use_container_width=True):
            st.session_state["next_page"] = "Fruit Quality Detector"
            st.rerun()

    if "next_page" in st.session_state:
        st.info(
            f"Choose **{st.session_state['next_page']}** in the sidebar "
            "to continue."
        )

    st.divider()
    st.markdown("### Project overview")
    a, b, c = st.columns(3)
    a.metric("Leaf classes", "38")
    b.metric("Fruit quality classes", "3")
    c.metric("Model architecture", "MobileNetV2")


# --------------------------------------------
# Leaf detector
# --------------------------------------------
elif page == "Leaf Disease Detector":
    st.title("🌱 Leaf Disease Detector")
    st.write(
        "Upload a clear image of a single leaf. The model predicts one of "
        "the 38 classes it learned from PlantVillage."
    )

    uploaded = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="leaf_upload",
    )

    if uploaded:
        image = Image.open(uploaded)
        show_prediction(
            image,
            "leaf_model.pt",
            "mobilenet_v2",
            "Leaf analysis",
            pretty_leaf_name,
        )

    st.info(
        "Predictions may be less reliable for unfamiliar crops, mixed "
        "backgrounds, poor lighting, or diseases absent from training."
    )


# --------------------------------------------
# Fruit detector
# --------------------------------------------
elif page == "Fruit Quality Detector":
    st.title("🍎 Fruit Quality Detector")
    st.write(
        "Upload a fruit image to classify it into one of the three "
        "FruitNet quality labels."
    )

    uploaded = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png", "webp"],
        key="fruit_upload",
    )

    if uploaded:
        image = Image.open(uploaded)
        show_prediction(
            image,
            "fruit_model.pt",
            "mobilenet_v2",
            "Fruit quality analysis",
            lambda name: name,
        )

    st.info(
        "This model predicts Good, Bad, or Mixed. It does not locate defects "
        "or identify a specific defect such as a bruise or fungal lesion."
    )


# --------------------------------------------
# Performance dashboard
# --------------------------------------------
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write(
        "Metrics below are recorded from the current training runs. "
        "Validation and test results are shown separately."
    )

    st.markdown("### Leaf disease model")
    a, b = st.columns(2)
    a.metric("Validation macro-F1", "99.04%")
    b.metric("Classes", "38")

    st.caption(
        "The leaf score is the best validation macro-F1 stored in the "
        "checkpoint. A held-out test result for this website's leaf model "
        "is not reported here."
    )

    st.divider()
    st.markdown("### Fruit quality model")
    a, b, c = st.columns(3)
    a.metric("Validation macro-F1", "99.37%")
    b.metric("Test accuracy", "99.59%")
    c.metric("Test macro-F1", "99.42%")

    st.caption(
        "Fruit results came from a random image-level split of the FruitNet "
        "dataset: 13,668 training, 2,929 validation, and 2,929 test images. "
        "Potential duplicates and external generalization have not yet been "
        "fully assessed."
    )

    st.markdown("### Fruit held-out test confusion matrix")
    st.write("Rows represent actual labels; columns represent predictions.")

    import pandas as pd

    cm = pd.DataFrame(
        [[1016, 2, 1], [7, 1740, 2], [0, 0, 161]],
        index=["Actual Bad", "Actual Good", "Actual Mixed"],
        columns=["Predicted Bad", "Predicted Good", "Predicted Mixed"],
    )
    st.dataframe(cm, use_container_width=True)

    st.warning(
        "High scores on a single dataset do not guarantee similar results "
        "on photos from farms, markets, different cameras, or new conditions."
    )


# --------------------------------------------
# About and limitations
# --------------------------------------------
elif page == "About & Limitations":
    st.title("ℹ️ About AgriVision AI")

    st.markdown("""
    AgriVision AI is a research prototype with two independent image
    classifiers:

    - **Leaf model:** MobileNetV2 trained for 38 PlantVillage classes.
    - **Fruit model:** MobileNetV2 trained for Good, Bad, and Mixed fruit
      quality labels from FruitNet.

    ### Important limitations

    - Predictions are model outputs, not professional diagnoses.
    - Softmax scores are not calibrated confidence probabilities.
    - The fruit model classifies quality labels; it does not localize defects.
    - Performance can change with lighting, camera quality, backgrounds,
      fruit varieties, and data distribution.
    - Dataset-level results require further duplicate checks and independent
      external testing.
    - Do not use this prototype alone for food safety or commercial grading.

    ### Responsible use

    Use the results as preliminary guidance. When a decision affects crop
    treatment, food safety, or commercial quality, seek appropriate expert
    assessment.
    """)
