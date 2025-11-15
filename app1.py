import streamlit as st
from groq import Groq
import pandas as pd
from PIL import Image

# Initialize Groq client (set your API key in Streamlit secrets or env variable)
client = Groq(api_key=st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else None)

# Define the prediction function using Groq LLM
def groq_predict(age, sex, bmi, children, smoker, region):
    """
    Use a Groq large language model to estimate insurance charges.
    """
    prompt = f"""
    You are an expert insurance data analyst.
    Estimate the health insurance charge for a person based on the following details:
    - Age: {age}
    - Sex: {sex}
    - BMI: {bmi}
    - Number of Children: {children}
    - Smoker: {smoker}
    - Region: {region}

    Respond with a single numeric estimate in NGN (₦) — no explanation, just the number.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Fast, reasoning-capable LLM
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    # Extract numeric estimate
    estimate = response.choices[0].message.content.strip()
    return estimate


# Streamlit App
def main():
    st.set_page_config(page_title="Insurance Charge Predictor (Groq LLM)", page_icon="💰", layout="centered")

    st.title("💰 Health Insurance Charge Predictor for people")
    st.markdown("This app uses a  estimate insurance costs based on user details.")
    st.divider()

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    st.divider()

    # Predict Button
    if st.button("🔮 Predict Insurance Charge"):
        with st.spinner("Estimating using Groq LLM..."):
            result = groq_predict(age, sex, bmi, children, smoker, region)
        st.success(f"💵 Estimated Charge: {result}")

    st.caption("⚙️ Powered by Groq LLM • Built with Streamlit")


if __name__ == "__main__":
    main()
