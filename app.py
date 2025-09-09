# app_pro_advanced.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model("skin_model.h5")

@st.cache_resource
def load_skin_type_model():
    return tf.keras.models.load_model("skin_type_model.h5")

@st.cache_resource
def load_lifestyle_model():
    with open("lifestyle_model.pkl", "rb") as f:
        return pickle.load(f)

skin_model = load_skin_model()
skin_type_model = load_skin_type_model()
lifestyle_model = load_lifestyle_model()

# ---------------------------
# App Layout
# ---------------------------
st.set_page_config(page_title="Skincare AI Pro", layout="wide")
st.title("🌟 Skincare AI Pro Assistant")

# Sidebar Inputs
st.sidebar.header("User Inputs")
name = st.sidebar.text_input("Your Name", "User")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
water = st.sidebar.slider("Daily Water Intake (liters)", 0, 5, 2)
uploaded_files = st.sidebar.file_uploader(
    "Upload Face Images (multiple allowed)", 
    type=["jpg","png","jpeg"], 
    accept_multiple_files=True
)

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_image(image):
    img_array = np.array(image.resize((224,224)))/255.0
    return np.expand_dims(img_array, axis=0)

def generate_skin_tip(score, skin_type, water, age):
    tips = []
    if score < 0.5:
        tips.append("Your skin health score is low. Follow a consistent skincare routine.")
    if water < 2:
        tips.append("Drink more water daily for healthy skin.")
    if age > 30:
        tips.append("Consider using anti-aging creams or serums.")
    if skin_type == "Oily":
        tips.append("Use oil-free moisturizers and clean your face regularly.")
    elif skin_type == "Dry":
        tips.append("Use hydrating creams and avoid harsh soaps.")
    else:
        tips.append("Maintain your balanced skin with regular care.")
    if not tips:
        tips.append("Your skin looks good! Keep up your healthy routine.")
    return tips

def generate_skincare_routine(skin_type):
    routine = []
    if skin_type == "Oily":
        routine = ["Cleanse twice daily", "Use oil-free moisturizer", "Apply sunscreen"]
    elif skin_type == "Dry":
        routine = ["Use gentle cleanser", "Apply hydrating moisturizer", "Use night cream"]
    elif skin_type == "Combination":
        routine = ["Cleanse daily", "Moisturize dry areas", "Use mattifying products on oily areas"]
    else:
        routine = ["Cleanse daily", "Moisturize", "Use sunscreen"]
    return routine

def create_pdf_report(name, df_scores, skin_types, routines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Skincare Report for {name}", ln=True, align="C")
    pdf.ln(10)
    
    for i, row in df_scores.iterrows():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{row['Image']}:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Score: {row['Score']}%", ln=True)
        pdf.cell(0, 10, f"Skin Type: {skin_types[i]}", ln=True)
        pdf.cell(0, 10, "Recommended Routine:", ln=True)
        for tip in routines[i]:
            pdf.cell(0, 8, f"- {tip}", ln=True)
        pdf.ln(5)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# ---------------------------
# Main App
# ---------------------------
if uploaded_files:
    st.subheader("Uploaded Images")
    col_count = min(3, len(uploaded_files))
    cols = st.columns(col_count)

    skin_scores = []
    skin_types = []
    routines = []

    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        cols[i % col_count].image(image, use_column_width=True, caption=f"Image {i+1}")
        img_array = preprocess_image(image)

        # Skin Health Score
        score = skin_model.predict(img_array)[0][0]
        skin_scores.append(round(score*100,2))

        # Skin Type
        type_pred = skin_type_model.predict(img_array)
        type_label = ["Oily","Dry","Normal","Combination"][np.argmax(type_pred)]
        skin_types.append(type_label)

        # Routine
        routine = generate_skincare_routine(type_label)
        routines.append(routine)

    # Display skin scores
    st.subheader("Skin Health Scores & Types")
    df_scores = pd.DataFrame({
        "Image": [f"Image {i+1}" for i in range(len(uploaded_files))],
        "Score": skin_scores,
        "Skin Type": skin_types
    })
    st.dataframe(df_scores)

    # Bar chart for scores
    st.subheader("Skin Score Comparison")
    fig, ax = plt.subplots()
    ax.bar(df_scores["Image"], df_scores["Score"], color="skyblue")
    ax.set_ylim(0,100)
    ax.set_ylabel("Skin Health Score (%)")
    st.pyplot(fig)

    # Personalized Tips
    st.subheader("💡 Personalized Skincare Tips")
    for i in range(len(uploaded_files)):
        st.write(f"**{df_scores['Image'][i]} ({skin_types[i]})**")
        tips = generate_skin_tip(skin_scores[i]/100, skin_types[i], water, age)
        for tip in tips:
            st.write(f"- {tip}")
        st.write("**Daily Skincare Routine:**")
        for r in routines[i]:
            st.write(f"- {r}")
        st.markdown("---")

    # Lifestyle Suggestions
    st.subheader("Lifestyle Suggestions")
    lifestyle_features = np.array([[age, water]])
    lifestyle_pred = lifestyle_model.predict(lifestyle_features)
    st.write(f"Predicted Suggestion: {lifestyle_pred[0]}")

    # PDF Export
    st.subheader("📄 Download Report")
    pdf_file = create_pdf_report(name, df_scores, skin_types, routines)
    st.download_button("Download PDF Report", data=pdf_file, file_name="skincare_report.pdf", mime="application/pdf")

else:
    st.info("Please upload at least one face image to get analysis.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit & AI models")
