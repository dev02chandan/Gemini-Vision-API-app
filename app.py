import streamlit as st
import google.generativeai as genai
import PIL.Image
import os

# Configure the Google API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Streamlit app title
st.title("Gemini Vision Object Detection")

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
        response = model.generate_content([custom_prompt, image])
        
        # Display the result
        st.write("Response:")
        st.write(response.text)
    else:
        st.write("Please upload an image first.")
