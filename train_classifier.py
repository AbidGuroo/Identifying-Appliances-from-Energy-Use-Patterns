import tensorflow as tf
import numpy as np
from sklearn.classifier_selection import train_test_split  # 隨機切分資料用
from tensorflow import keras
from keras import regularizers
from keras.callbacks import classifierCheckpoint


# Load Numpy Arrays
x = np.load("x_train.npy")
y = np.load("y_train.npy")

# Perform Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Scale/Normalize
x_train = x_train / 255
x_test = x_test / 255


# One Hot Encoding
NUM_CLASSES = 11
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

## Initialize the Classifier
classifier = keras.Sequential()

## Network Definition
classifier.add(
    keras.layers.Conv2D(
        32, activation=tf.nn.relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]
    )
)
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(64, activation=tf.nn.relu, kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(128, activation=tf.nn.relu, kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(256, activation=tf.nn.relu, kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(512, activation=tf.nn.relu, kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Flatten())
classifier.add(keras.layers.Dense(550, activation=tf.nn.relu))  # 普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(330, activation=tf.nn.relu))  # 普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(110, activation=tf.nn.relu))  # 普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(44, activation=tf.nn.relu))  # 普通的全連接層
classifier.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))
classifier.summary()


# Compile the Classifier
classifier.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

#
weightfilepath = "weights.best.hdf5"

# Create Best Weights Checkpoint
checkpoint = classifierCheckpoint(
    weightfilepath, 
    monitor="val_acc", 
    verbose=1, 
    save_best_only=True, 
    mode="max"
)
callbacks_list = [checkpoint]

# Fit the classifier
train = classifier.fit(
    x_train,
    y_train,
    callbacks=callbacks_list,
    validation_data=(x_test, y_test),
    epochs=1000,
)
