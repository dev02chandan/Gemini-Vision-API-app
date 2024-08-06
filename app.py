import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
import tempfile
import time

# Configure the Google API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

st.image('logo.png', width=200)

# Streamlit app title
st.title("Vision LLM")

# Option to choose between image or video
media_type = st.radio("Choose media type:", ("Image", "Video"))

# Upload media
if media_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Text input for custom prompt
custom_prompt = st.text_input("Enter your custom prompt:", "Describe this media.")

# Generate Response button
if st.button("Generate Response"):
    if uploaded_file is not None:
        if media_type == "Image":
            # Display the uploaded image
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Load and process the image
            st.write("Processing the image...")
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

            # Generate content with the model
            response = model.generate_content([custom_prompt, image],
                                              safety_settings={
                # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
            })

            # Display the result
            st.write("Response:")
            st.write(response.text)
        
        elif media_type == "Video":
            # Save the uploaded video to a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_file_name = tfile.name

            st.write("Uploading file...")

            # Upload video to Google Generative AI
            video_file = genai.upload_file(path=video_file_name, mime_type="video/mp4")
            st.write("Processing video...")

            # Check processing state
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                st.write("Failed to process video.")
                raise ValueError(video_file.state.name)
            
            # Create the prompt
            prompt = custom_prompt

            # Set the model to Gemini 1.5 Pro
            model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

            # Make the LLM request
            st.write("Making LLM inference request...")
            response = model.generate_content([prompt, video_file],
                                              request_options={"timeout": 600})
            
            # Display the result
            st.write("Response:")
            st.write(response.text)
            
            # Delete the uploaded file
            genai.delete_file(video_file.name)
    else:
        st.write(f"Please upload a {media_type.lower()} first.")
