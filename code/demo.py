import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import streamlit as st
from ultralytics import YOLO
from model.resnet50 import resnet50  # Ensure this import is correct
import io
import cv2
import numpy as np

# -------------------
# Set Page Configuration
# -------------------
st.set_page_config(
    page_title="YOLO Detection + ResNet Classification",
    layout="wide"
)

# -------------------
# Configuration
# -------------------

# Class names for ResNet classification
class_names = ["mild", "moderate", "severe", "very severe"]

# Paths to models and fonts
DETECT_MODEL_PATH = "/media/abdul/New Volume1/inventra/yoloo/runs/detect/acne_detection3/weights/best.pt"
RESNET_MODEL_PATH = "/media/abdul/New Volume1/inventra/LDL/code/logs/best_model.pth"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Replace with a valid font path
CASCADE_PATH = "cascades/haarcascade_frontalface_default.xml"  # Path to Haar Cascade XML

# -------------------
# Load Models
# -------------------

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_resnet_model(model_path):
    model = resnet50()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_face_cascade(cascade_path):
    if not os.path.exists(cascade_path):
        st.error(f"Haar Cascade XML file not found at {cascade_path}. Please ensure the path is correct.")
        return None
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

# Load YOLO model
yolo_model = load_yolo_model(DETECT_MODEL_PATH)

# Load ResNet model
resnet_model = load_resnet_model(RESNET_MODEL_PATH)

# Load Haar Cascade for face detection
face_cascade = load_face_cascade(CASCADE_PATH)

# -------------------
# Define Transforms
# -------------------

normalize = transforms.Normalize(
    mean=[0.45815152, 0.361242, 0.29348266],
    std=[0.2814769, 0.226306, 0.20132513]
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# -------------------
# Helper Functions
# -------------------

def detect_face(image, face_cascade):
    """
    Detects the largest face in the image and returns the cropped face.
    """
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("No faces detected in the image. Please try a different image.")
        return None  # No face detected
    
    # Log number of faces detected
    st.write(f"Detected {len(faces)} face(s).")
    
    # Draw rectangles around detected faces for debugging
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert back to PIL Image and display
    debug_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    st.image(debug_image, caption='Face Detection Debug Image', use_container_width=True)
    
    # Select the largest face detected
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    (x, y, w, h) = largest_face
    
    # Crop the face from the image
    cropped_face = image.crop((x, y, x + w, y + h))
    return cropped_face

def brighten_image(image, factor=1.5):
    """
    Brightens the image by the given factor.
    """
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)
    return brightened_image

def detect_and_classify_with_face(image, yolo_model, resnet_model, face_cascade):
    """
    Detects and crops the face, brightens the image, performs YOLO detection,
    and ResNet classification. Draws bounding boxes labeled 'acne' on the face.
    """
    # -------- Face Detection and Cropping --------
    if face_cascade is None:
        st.error("Face cascade classifier not loaded.")
        return image, "Face detection failed."
    
    cropped_face = detect_face(image, face_cascade)
    
    if cropped_face is None:
        st.warning("Face detection failed.")
        return image, "Face detection failed."
    
    # -------- Image Brightening --------
    brightened_image = brighten_image(cropped_face, factor=1.5)  # Adjust the factor as needed
    
    # Display the brightened image
    #st.image(brightened_image, caption='Brightened Face', use_container_width=True)
    
    # -------- ResNet Classification --------
    # Transform the input image for ResNet
    input_tensor = transform(brightened_image).unsqueeze(0)  # Add batch dimension

    # Classify the entire image with ResNet
    with torch.no_grad():
        _, _, out3 = resnet_model(input_tensor)
    probs = F.softmax(out3, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    pred_label = class_names[pred_idx]
    classification_text = f"**Classification:** {pred_label}"
    
    # -------- YOLO Detection --------
    # Run YOLO detection on the brightened face image
    results = yolo_model(brightened_image, conf=0.08, iou=0.07)  # Adjust 'conf' and 'iou' as needed

    # Clone the brightened face image for drawing bounding boxes
    output_image = brightened_image.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Load a TrueType font
    try:
        font = ImageFont.truetype(FONT_PATH, size=16)
    except IOError:
        font = ImageFont.load_default()
        st.warning("Custom font not found. Using default font.")
    
    # Draw bounding boxes labeled 'acne' from YOLO detections
    for detection in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        label = "acne"

        # Draw the bounding box on the output image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Add label text above the bounding box
        text = label

        # Calculate text size using getbbox
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_background = [x1, y1 - text_height, x1 + text_width, y1]
        draw.rectangle(text_background, fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    return output_image, classification_text

def detect_and_classify_without_face(image, yolo_model, resnet_model):
    """
    Processes the entire image without face detection. Performs image brightening,
    YOLO detection, and ResNet classification.
    """
    # -------- Image Brightening --------
    brightened_image = brighten_image(image, factor=1)  # Adjust the factor as needed
    
    # Display the brightened image
    #st.image(brightened_image, caption='Brightened Image', use_container_width=True)
    
    # -------- ResNet Classification --------
    # Transform the input image for ResNet
    input_tensor = transform(brightened_image).unsqueeze(0)  # Add batch dimension

    # Classify the entire image with ResNet
    with torch.no_grad():
        _, _, out3 = resnet_model(input_tensor)
    probs = F.softmax(out3, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    pred_label = class_names[pred_idx]
    classification_text = f"**Classification:** {pred_label}"
    
    # -------- YOLO Detection --------
    # Run YOLO detection on the brightened image
    results = yolo_model(brightened_image, conf=0.08, iou=0.07)  # Adjust 'conf' and 'iou' as needed

    # Clone the brightened image for drawing bounding boxes
    output_image = brightened_image.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Load a TrueType font
    try:
        font = ImageFont.truetype(FONT_PATH, size=16)
    except IOError:
        font = ImageFont.load_default()
        st.warning("Custom font not found. Using default font.")
    
    # Draw bounding boxes labeled 'acne' from YOLO detections
    for detection in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        label = "acne"

        # Draw the bounding box on the output image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Add label text above the bounding box
        text = label

        # Calculate text size using getbbox
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_background = [x1, y1 - text_height, x1 + text_width, y1]
        draw.rectangle(text_background, fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    return output_image, classification_text

# -------------------
# Streamlit App
# -------------------

def main():
    st.title("YOLO Detection + ResNet Classification")

    st.write(
        """
        **Instructions:**
        - Choose an input method: Upload an image or use your webcam.
        - If you choose **Use Webcam**, the app will detect and crop the face, brighten the image, perform YOLO object detection, and ResNet classification.
        - Bounding boxes labeled 'acne' will be drawn around detected acne regions (only for webcam input).
        - If you choose **Upload Image**, the app will brighten the entire image, perform YOLO object detection, and ResNet classification without face detection.
        - The overall classification result will be displayed alongside the image.
        """
    )

    # Select input method
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

    image = None

    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Open the image
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
    else:
        # Webcam input
        webcam_image = st.camera_input("Take a picture")
        if webcam_image is not None:
            # Open the image
            try:
                image = Image.open(webcam_image).convert("RGB")
                st.image(image, caption='Captured Image', use_container_width=True)  # Updated parameter
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return

    # Proceed if image is loaded
    if image is not None:
        # Perform detection and classification based on input method
        with st.spinner("Processing..."):
            if input_method == "Use Webcam":
                output_image, classification_text = detect_and_classify_with_face(image, yolo_model, resnet_model, face_cascade)
            else:
                output_image, classification_text = detect_and_classify_without_face(image, yolo_model, resnet_model)
        
        # Display the output image and classification side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(output_image, caption='Processed Image', use_container_width=True)  # Updated parameter

        with col2:
            st.markdown(classification_text)
            st.write("")

            # Convert PIL image to bytes for download
            buf = io.BytesIO()
            output_image.save(buf, format='PNG')
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
