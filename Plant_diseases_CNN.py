import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ========================
# Configuration Parameters
# ========================
DATASET_PATH = r'D:\Suresh\Multi_disease_detection\MangoLeafBD Dataset'  # <-- Update path if needed
IMAGE_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 32
MODEL_SAVE_PATH = "Mango_leaf_disease_model.h5"

# ========================
# Load and preprocess data
# ========================
def load_data(dataset_path):
    data, labels = [], []
    class_names = sorted(os.listdir(dataset_path))  # Sorted for consistency

    for idx, class_name in enumerate(class_names):
        folder_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(folder_path):
            continue
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels), class_names

# ========================
# Prepare the dataset
# ========================
print("Loading and preprocessing dataset...")
X, y, class_names = load_data(DATASET_PATH)
X = X.astype('float32') / 255.0  # Normalize
y = to_categorical(y, num_classes=len(class_names))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# Data Augmentation
# ========================
datagen = ImageDataGenerator(rotation_range=20,
                             zoom_range=0.2,
                             horizontal_flip=True)
datagen.fit(X_train)

# ========================
# CNN Model Definition
# ========================
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ========================
# Compile & Train the Model
# ========================
model = build_model(len(class_names))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Training the model...")
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS)

# ========================
# Save the model
# ========================
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ========================
# Plot Training History
# ========================
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# Evaluation
# ========================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ========================
# Prediction Function
# ========================
def predict_image(image_path, model, class_names):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# ========================
# Predict from CLI argument
# ========================
if len(sys.argv) > 1:
    predict_image_path = sys.argv[1]
    if os.path.exists(predict_image_path):
        print(f"\nPredicting for: {predict_image_path}")
        model = load_model(MODEL_SAVE_PATH)
        predicted_class, confidence = predict_image(predict_image_path, model, class_names)
        if predicted_class:
            print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
        else:
            print("Prediction failed.")
    else:
        print(f"Image path '{predict_image_path}' does not exist.")