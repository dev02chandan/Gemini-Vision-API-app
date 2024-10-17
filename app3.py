import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
import tempfile
import time
from PIL import ImageDraw, ImageFont
import json

# Configure the Google API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

st.image("logo.png", width=300)

# Streamlit app title
st.title("Bounding Boxes")

# Option to choose between image or video
# media_type = st.radio("Choose media type:", ("Image"))

generation_config = {
    "response_mime_type": "application/json",
}

# Option to choose between the gemini models
model_choice = st.selectbox("Choose the model:", ("Model F", "Model P"))

model_choice = "gemini-1.5-Flash" if model_choice == "Model F" else "gemini-1.5-Pro"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Text input for custom prompt

custom_prompt = (
    "Return a bounding box for all objects that you see in JSON format. "
    "If there are multiple instances of the same object, include each one. "
    "The format should be as follows:\n"
    "{\n"
    "  'objects': [\n"
    "    {\n"
    "      'label': 'object_name',\n"
    "      'bounding_box': [ymin, xmin, ymax, xmax]\n"
    "    },\n"
    "    {\n"
    "      'label': 'object_name',\n"
    "      'bounding_box': [ymin, xmin, ymax, xmax]\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}"
)


# Generate Response button
if st.button("Generate Response"):
    if uploaded_file is not None:
        # Display the uploaded image
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Load and process the image
        st.write("Processing the image...")
        model = genai.GenerativeModel(
            model_name=f"models/{model_choice.lower()}",
            generation_config=generation_config,
        )

        # Generate content with the model
        response = model.generate_content(
            [custom_prompt, image],
            safety_settings={
                # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
            },
        )

        # Display the response text
        st.write("Response:")
        # st.write(response.text)

        # Load the JSON output from the response
        json_output = json.loads(response.text)

        image = PIL.Image.open(uploaded_file)
        original_width, original_height = image.size  # Get original dimensions
        draw = ImageDraw.Draw(image)

        # Draw bounding boxes with scaling
        for obj in json_output["objects"]:
            label = obj["label"]
            # Scale the bounding box coordinates
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(xmin / 1000 * original_width)  # Scale x
            ymin = int(ymin / 1000 * original_height)  # Scale y
            xmax = int(xmax / 1000 * original_width)  # Scale x
            ymax = int(ymax / 1000 * original_height)  # Scale y

            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text(
                (xmin, ymin),
                label,
                fill="red",
                font=ImageFont.truetype("Arial.ttf", 16),
            )

        # Display the image with bounding boxes
        st.image(
            image,
            caption="Uploaded Image with Bounding Boxes",
            use_column_width=True,
        )
