import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = 128
MODEL_PATH = "C:\Suresh\Multi_disease_detection 95% code\Mango_leaf_disease_model_1.h5"
CLASS_NAMES_PATH = "class_names.json"

# Load model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        index = np.argmax(prediction)
        return class_names[index]
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide an image path.")
        sys.exit()
    
    image_path = sys.argv[1]
    result = predict_image(image_path)
    print(result)
