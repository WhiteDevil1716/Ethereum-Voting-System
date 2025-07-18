import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# Set paths for training and test data
train_img_folder = 'C:/Users/rithv/Retinal Disease Detection/training/images'
train_mask_folder = 'C:/Users/rithv/Retinal Disease Detection/training/masks'
test_img_folder = 'C:/Users/rithv/Retinal Disease Detection/test/images'

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Load training and testing images
train_images, train_filenames = load_images_from_folder(train_img_folder)
test_images, test_filenames = load_images_from_folder(test_img_folder)

# Normalize images by resizing and scaling pixel values to [0, 1]
def preprocess_images(images, target_size=(128, 128)):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)  # Resize to target size
        img_normalized = img_resized / 255.0  # Normalize pixel values
        processed_images.append(img_normalized)
    return np.array(processed_images)

# Preprocess training and test images
X_train = preprocess_images(train_images)
X_test = preprocess_images(test_images)

# Load masks and preprocess them (binary masks)
def load_and_preprocess_masks(mask_folder, filenames, target_size=(128, 128)):
    masks = []
    for filename in filenames:
        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_resized = cv2.resize(mask, target_size)
            mask_binary = mask_resized // 255  # Convert to binary mask (0 or 1)
            masks.append(mask_binary)
    return np.array(masks)

# Load and preprocess training masks
y_train = load_and_preprocess_masks(train_mask_folder, train_filenames)
y_train = np.expand_dims(y_train, axis=-1)  # Add channel dimension

# For the test set, we won't have masks, but we can assume ground truth if available
# For demonstration, let's assume test labels are also available
# This can be modified based on your dataset
y_test = np.expand_dims(np.random.randint(0, 2, size=len(X_test)), axis=-1)  # Random labels for testing

# Build a simple CNN model for classification
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 128, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Classification Report
print(classification_report(y_test, y_pred))

# Visualize some test images and predictions
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'Pred: {y_pred[i][0]}')
    plt.axis('off')
plt.show()
