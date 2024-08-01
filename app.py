import streamlit as st
import google.generativeai as genai
import PIL.Image
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

# Configure the Google API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

st.image('logo.png', width=200)

# Streamlit app title
st.title("Vision LLM")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Text input for custom prompt
custom_prompt = st.text_input("Enter your custom prompt:", "What is in this photo?")

# Generate Response button
if st.button("Generate Response"):
    if uploaded_file is not None:
        # Display the uploaded image
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Load and process the image
        st.write("Processing the image...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        # Generate content with the model
        response = model.generate_content([custom_prompt, image],
                                          safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    })
        
        # Display the result
        st.write("Response:")
        st.write(response.text)
    else:
        st.write("Please upload an image first.")
