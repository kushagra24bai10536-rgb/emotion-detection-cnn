import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                      Dropout, Flatten, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Paths (FER2013 dataset chahiye)
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
MODEL_PATH = "model/emotion_model.h5"

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train():
    os.makedirs("model", exist_ok=True)

    # Data augmentation
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()