import streamlit as st
import numpy as np
from PIL import Image
import random

labels = ["acne", "clear", "dark_spots", "wrinkles"]

st.set_page_config(page_title="AI Skincare Coach", page_icon="🧴")
st.title("🧴 AI Skincare + Lifestyle Coach")
st.write("Upload your face photo + lifestyle info, and get skin analysis & risk prediction.")

uploaded_file = st.file_uploader("📷 Upload your face photo", type=["jpg", "jpeg", "png"])

sleep = st.slider("😴 Sleep hours", 3, 10, 7)
water = st.slider("💧 Water intake (litres)", 0, 5, 2)
junk = st.selectbox("🍔 Junk food today?", [0, 1])  
stress = st.slider("😓 Stress level", 1, 3, 1)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Face", use_column_width=True)

    skin_condition = random.choice(labels)

    risk_score = 0
    if sleep < 5: risk_score += 1
    if water < 2: risk_score += 1
    if junk == 1: risk_score += 1
    if stress > 2: risk_score += 1
    risk = "🔥 High" if risk_score >= 2 else "✅ Low"

    st.subheader("📊 Results")
    st.write(f"**Skin Condition Detected:** {skin_condition}")
    st.write(f"**Acne Risk Prediction:** {risk}")

    st.subheader("💡 Suggestions")
    if skin_condition == "acne":
        st.warning("Wash face twice daily + try salicylic acid cleanser.")
    elif skin_condition == "dark_spots":
        st.info("Use Vitamin C serum & sunscreen daily.")
    elif skin_condition == "wrinkles":
        st.success("Stay hydrated + apply Retinol at night.")
    else:
        st.balloons()
        st.success("Great skin! Keep maintaining your routine.")

    if risk == "🔥 High":
        st.error("⚠️ Improve your lifestyle: Sleep more, drink water, reduce junk food.")
    else:
        st.success("👍 Keep up the healthy lifestyle!")
