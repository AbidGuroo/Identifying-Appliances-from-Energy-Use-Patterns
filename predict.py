## Run Predictions using Trained KEras Classifier
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator     #導入kearas圖像預處理模組
from sklearn.classifier_selection import train_test_split #隨機切分資料用
from PIL import Image 
import os 
import csv


## Initialize the Model
classifier = keras.Sequential()

classifier.add(keras.layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(3, 3), input_shape=(256,118,3)))  #新增卷積層
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))   #最大池化
classifier.add(keras.layers.Dropout(0.25))                 #防止過擬合

classifier.add(keras.layers.Conv2D(64, activation=tf.nn.relu,kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(128, activation=tf.nn.relu,kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(256, activation=tf.nn.relu,kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Conv2D(512, activation=tf.nn.relu,kernel_size=(3, 3)))
classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.25))

classifier.add(keras.layers.Flatten())
classifier.add(keras.layers.Dense(550, activation=tf.nn.relu))  #普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(330, activation=tf.nn.relu))  #普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(110, activation=tf.nn.relu))  #普通的全連接層
classifier.add(keras.layers.Dropout(0.25))
classifier.add(keras.layers.Dense(44, activation=tf.nn.relu))  #普通的全連接層
classifier.add(keras.layers.Dense(11, activation=tf.nn.softmax))
classifier.summary()

classifier.load_weights('weights.best.hdf5')

# Compile the Classifier
classifier.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

router = './data/testdata'


with open('./data/submission.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        try:
            img = Image.open('{}/{}.png'.format(router,row[0]))
            img = np.array(img)
            img = img.astype('float32')
            img = img / 255
            img = img.reshape(1, 256, 118, 3)
            results = np.argmax(classifier.predict(img)) #預測
            confidence = np.max(classifier.predict(img))*100  #信心指數
            confidence = '%.2f' % confidence
            print(results)
            print(confidence)
        except:
            continue
