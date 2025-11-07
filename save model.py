import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Set dataset path (make sure to update this)
DATASET_PATH = r'C:\\PERSONAL\\Priya\\Final_year_project\\Multi_disease_detection\\MangoLeafBD Dataset'

IMAGE_SIZE = 128
EPOCHS = 50
BATCH_SIZE = 32

# Load and preprocess dataset
def load_data(dataset_path):
    data = []
    labels = []
    classes = os.listdir(dataset_path)

    for idx, plant_class in enumerate(classes):
        folder_path = os.path.join(dataset_path, plant_class)
        if not os.path.isdir(folder_path):
            continue
        for image_file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                data.append(img)
                labels.append(idx)
            except:
                pass

    return np.array(data), np.array(labels), classes

print("🔄 Loading dataset...")
X, y, class_names = load_data(DATASET_PATH)
X = X / 255.0
y = to_categorical(y)

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)

# Build CNN
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
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
print("🚀 Training model...")
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS)

# Save model
model.save("Mango_leaf_disease_model_1.h5")
print("✅ Model saved as 'Mango_leaf_disease_model_1.h5'")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training History")
plt.show()
