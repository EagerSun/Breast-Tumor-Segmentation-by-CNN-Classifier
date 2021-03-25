from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class Model():
    def train(self, input_train, label_train, input_test, label_test):
        # encode and standardize
        label_trainHot = tf.keras.utils.to_categorical(label_train, num_classes = 2)
        label_testHot = tf.keras.utils.to_categorical(label_test, num_classes = 2)
        print(label_train.shape)
        # define shape
        shape = (50, 50, 3)
        # create model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=shape, name='conv_l1', strides=1))
        model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_l1'))
        model.add(Dropout(0.25, name='dropout_l1'))
        model.add(Conv2D(64, (3, 3), activation='relu', name='conv_l2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_l2'))
        model.add(Dropout(0.25, name='dropout_l2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(128, activation='relu', name='dense_l1'))
        model.add(Dropout(0.25, name='dropout_l3'))
        # number of classes is 2
        model.add(Dense(2, activation='softmax', name='dense_l2'))
        # show summary
        model.summary()
        # pip install pydot 
        # pip install graphviz
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # now we fit and evaluate the data
        model.fit(input_train, label_trainHot, validation_data=(input_test, label_testHot), epochs=20, batch_size=64, verbose=2)
        model.save(os.path.join(os.getcwd(), 'model.h5'));