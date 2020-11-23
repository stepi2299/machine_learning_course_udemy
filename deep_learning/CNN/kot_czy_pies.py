import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

data_dir = 'dataset/kagglecatsanddogs_3367a/PetImages'
img_size = 128
batch_size = 32

train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=0,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb'
)
val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=0,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb'
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1/255, input_shape=(img_size, img_size, 3)))
# Convolutional
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))
# Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# 2 convolutional and pooling
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# flattening
model.add(tf.keras.layers.Flatten())
# Full connection
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit(x=train, epochs=10, validation_data=val)

# Evaluation
print(model.evaluate(train))
print(model.evaluate(val))
