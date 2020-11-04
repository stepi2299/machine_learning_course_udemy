import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import numpy as np


# Image Preprocessing
# geometric trasnformations are to avoid overfitting (so rotation itp to modify) - image agmentation

# preprocessing training set
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  # here is also feature scaling (rescaling)
training_set = train_data_gen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')  # implementing dataset

# preprocessing test set
test_data_gen = ImageDataGenerator(rescale=1./255)
test_set = test_data_gen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Building the CNN
cnn = tf.keras.models.Sequential()

# convolutional
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2 convolutional and pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN
# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Make a single prediction
test_image = image.load_img('dataset/single_prediction/baksio.jpg', target_size=(64, 64))  # importing pln image
test_image = image.img_to_array(test_image)  # converting image into numpy array format
test_image = np.expand_dims(test_image, axis=0)  # adding extra dimension which correspond to the batch and which will contain image in a batch
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:  # because result is in the batch dimension (first dimension is number of batch and other is order in batch)
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
