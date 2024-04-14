import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

@st.cache(allow_output_mutation=True)
def get_predictor_model():
    from model import Model
    model = Model()
    return model

header = st.container()
model = get_predictor_model()

with header:
    st.title('Hello!')
    st.text('Using this app you can classify whether there is a fight on a street, fire, car crash, or if everything is okay.')

uploaded_file = st.file_uploader("Choose an image or video file...")

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image/"):
        # Handle image file
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        label_text = model.predict(image=image_np)['label'].title()
        st.write(f'Predicted label for the image is: **{label_text}**')
        st.image(image_np)

    elif file_type.startswith("video/"):
        # Handle video file
        st.write("Uploaded file is a video.")
        
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_file.getvalue())
            video_file_path = temp_video_file.name
        
        # Use OpenCV to read the video
        video_capture = cv2.VideoCapture(video_file_path)

        if not video_capture.isOpened():
            st.error("Could not open video file. Please try a different file.")
        else:
            # Process video frames
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if not ret:
                    break  # Exit the loop when no more frames are available

                # Convert the frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use the model to predict the frame
                label_text = model.predict(image=frame_rgb)['label'].title()
                
                st.write(f'Predicted label for the current frame: **{label_text}**')
                st.image(frame_rgb)
            
            video_capture.release()
        
        # Clean up the temporary file
        os.remove(video_file_path)
