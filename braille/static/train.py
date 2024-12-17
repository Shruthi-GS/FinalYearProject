# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Dataset loading and preprocessing
def load_dataset(data_dir, class_labels):
    images, labels = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(data_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            label_char = filename[0].upper()
            labels.append(class_labels[label_char])
    images = np.array(images) / 255.0  # Normalize images
    labels = to_categorical(np.array(labels), num_classes=len(class_labels))
    return images, labels


# CNN Model Definition
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
    model.save('braille_cnn_model.h5')
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Confusion Matrix and Classification Report
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[chr(i) for i in range(65, 91)],
                yticklabels=[chr(i) for i in range(65, 91)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=[chr(i) for i in range(65, 91)]))


# Predict new Braille character
def predict_braille(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    classes = [chr(i) for i in range(65, 91)]
    return classes[np.argmax(prediction)]


# if __name__ == "__main__":
#     # Paths and Class Labels
#     data_dir = 'dataset'
#     class_labels = {chr(i): i - 65 for i in range(65, 91)}  # A-Z mapping to 0-25

#     # Load and split dataset
#     images, labels = load_dataset(data_dir, class_labels)
#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

#     # Model creation and training
#     model = create_model()
#     train_model(model, X_train, y_train, X_test, y_test)

#     # Model evaluation
#     evaluate_model(model, X_test, y_test)

#     # Predict a new image
#     braille_char = predict_braille('braille_sentence.jpg', load_model('braille_cnn_model.h5'))
#     print(f"Predicted Braille Character: {braille_char}")


def evaluate_only():
    # Evaluate model
    print("Loading model...")
    model = load_model('braille_cnn_model.h5')  # Load pre-trained model
    _, X_test, _, y_test = load_and_split_dataset(data_dir, class_labels)  # Only load test data
    evaluate_model(model, X_test, y_test)  # Assume evaluate_model is defined elsewhere

    # Predict a new image
    print("Predicting Braille character from an image...")
    braille_char = predict_braille('braille_sentence.jpg', model)  # Assume predict_braille is defined elsewhere
    print(f"Predicted Braille Character: {braille_char}")

if __name__ == '__main__':
    # To run only evaluation:
    evaluate_only()