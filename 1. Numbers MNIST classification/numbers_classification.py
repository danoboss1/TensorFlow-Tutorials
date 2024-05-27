# inspired by official TensorFlow tutorial on clothes MNIST
# https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf

import numpy as np  
import matplotlib.pyplot as plt

# predefined special/magic attribute 
print(tf.__version__)

number_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = number_mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure()
plt.imshow(train_images[7])
plt.colorbar()
plt.grid(False)
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape input to include channel dimension
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    tf.keras.layers.Flatten(),  # Flatten layer
    tf.keras.layers.Dense(10)  # Output layer with 10 units (one for each digit class)
])

# An optimizer is used to adjust the model's parameters during training to minimize the loss function and improve its performance
# A loss function quantifies the difference between a model's predictions and the actual target values
# Metrics is rating according to which you decide
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("Final test accuracy: ", test_acc)