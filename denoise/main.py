import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("Image Denoising App")

# Function to load an image
def load_image():
    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if file is not None:
        image = Image.open(file)
        return np.array(image)
    else:
        return None

# Function to add Gaussian noise
def add_gaussian_noise(image, mean, variance):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, variance ** 0.5, (row, col, ch)).astype(np.float32)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure values are within [0, 1]
    return noisy_image

# Function to add salt and pepper noise
def add_salt_and_pepper_noise(image, prob):
    row, col, ch = image.shape
    noisy_image = np.copy(image)
    for i in range(row):
        for j in range(col):
            rand = np.random.random()
            if rand < prob:
                noisy_image[i][j] = 0  # Pepper
            elif rand > 1 - prob:
                noisy_image[i][j] = 1  # Salt
    return noisy_image

# Function to apply mean filter
def apply_mean_filter(image):
    return cv2.blur(image, (3, 3))

# Function to apply min filter
def apply_min_filter(image):
    return cv2.erode(image, np.ones((3, 3), np.uint8))

# Function to apply max filter
def apply_max_filter(image):
    return cv2.dilate(image, np.ones((3, 3), np.uint8))

# Function to apply median filter
def apply_median_filter(image):
    return cv2.medianBlur((image * 255).astype(np.uint8), 3) / 255.0

# Load the image
image = load_image()

if image is not None:
    image = image.astype(np.float32) / 255.0  # Normalize image data to the range [0.0, 1.0]
    st.image(image, caption="Original Image", use_column_width='always')

    # Sidebar options
    st.sidebar.header("Noise Options")
    noise_type = st.sidebar.selectbox("Select noise type", ["None", "Gaussian", "Salt and Pepper"])
    if noise_type == "Gaussian":
        mean = st.sidebar.slider("Mean", -0.5, 0.5, 0.0)
        variance = st.sidebar.slider("Variance", 0.0, 0.1, 0.01)
        noisy_image = add_gaussian_noise(image, mean, variance)
        #print(noisy_image[0:10,0:10])
    elif noise_type == "Salt and Pepper":
        prob = st.sidebar.slider("Probability", 0.0, 1.0, 0.05)
        noisy_image = add_salt_and_pepper_noise(image, prob)
        #print(noisy_image[0:10,0:10])

    else:
        noisy_image = image

    st.image(noisy_image, caption="Noisy Image", use_column_width='always')

    st.sidebar.header("Denoising Filters")
    filter_type = st.sidebar.selectbox("Select filter", ["None", "Mean", "Min", "Max", "Median"])
    if filter_type == "Mean":
        denoised_image = apply_mean_filter(noisy_image)
    elif filter_type == "Min":
        denoised_image = apply_min_filter(noisy_image)
    elif filter_type == "Max":
        denoised_image = apply_max_filter(noisy_image)
    elif filter_type == "Median":
        denoised_image = apply_median_filter(noisy_image)
    else:
        denoised_image = noisy_image

    st.image(denoised_image, caption="Denoised Image", use_column_width='always')

else:
    st.text("Please upload an image.")
