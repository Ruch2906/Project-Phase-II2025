# Updated model_CNN.py to match your folder structure and fix image loading issues

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Input image dimensions
img_rows, img_cols = 64, 64
img_channels = 1

# Folder paths (each class should be a subfolder like 0, 1, 2, 3)
path1 = "C:/Users/soura/Downloads/finalyr project/Code Cricket shot/Code Cricket shot/training"
path2 = "C:/Users/soura/Downloads/finalyr project/Code Cricket shot/Code Cricket shot/testing"

# Helper to load images from a folder

def load_images_from_folder(base_path):
    images = []
    labels = []
    class_folders = sorted(os.listdir(base_path))
    print(f"Classes found in {base_path}: {class_folders}")
    
    for label, class_folder in enumerate(class_folders):
        folder_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(folder_path, file)
                    img = Image.open(img_path).convert('L').resize((img_rows, img_cols))
                    img_array = np.array(img, dtype='float32')
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training data
X_train, y_train = load_images_from_folder(path1)
X_test, y_test = load_images_from_folder(path2)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("No valid images were loaded from the training or testing folders.")

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train /= 255.0
X_test /= 255.0

# Encode labels
nb_classes = len(np.unique(y_train))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model setup
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
csv_logger = CSVLogger('training_log.csv')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train
history = model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, Y_test),
    callbacks=[csv_logger, early_stopping, checkpoint],
    verbose=1
)

# Save model
model.save('final_model.h5')
model.save_weights('cnn_weights.h5')

# Evaluate
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(np.argmax(Y_test, axis=1), y_pred))
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))

# Plotting
plt.figure(1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.figure(2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()