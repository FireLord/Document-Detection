import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained HED model
hed_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)

def preprocess_image(image_path):
    # Load and preprocess the image for the HED model
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read image from {image_path}")

    image = cv2.resize(image, (600, 600))
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image

def edge_detection(image):
    # Use the HED model for edge detection
    edges = hed_model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
    return edges

# Example usage
image_path = 'IMG_20231213_152313.jpg'
try:
    input_image = preprocess_image(image_path)
    edges = edge_detection(input_image)

    # Apply a threshold to obtain a binary edge mask
    threshold_value = 0.1  # Adjust this threshold value as needed
    edge_mask = (edges > threshold_value).astype(np.uint8) * 255

    # Resize the edge mask to match the dimensions of the original image
    edge_mask_resized = cv2.resize(edge_mask, (input_image.shape[1], input_image.shape[0]))

    # Convert the original image to RGB
    original_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Mark the edges with green color
    marked_image = cv2.addWeighted(original_image.astype(np.uint8), 1, cv2.merge([edge_mask_resized, np.zeros_like(edge_mask_resized), np.zeros_like(edge_mask_resized)]).astype(np.uint8), 0.5, 0)

    # Display the results
    plt.imshow(marked_image)
    plt.title('Detected Edges (Green)')
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
