import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers

import random
import numpy as np


from .utils import sample_normalization

def dataaugmentation(X):
    X_aug = X
    X_aug = tf.keras.layers.RandomFlip('horizontal')(X_aug)
    X_aug = tf.keras.layers.RandomCrop(448, 448)(X_aug)
    X_aug = tf.keras.layers.Resizing(512, 512)(X_aug)
    return X_aug

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.X_list) // self.batch_size

    def flow(self,
             X_list, y_list, epochs=1, batch_size=32, shuffle=True):
        self.batch_size = batch_size

        c = list(zip(X_list, y_list))
        if shuffle:
            random.shuffle(c)
        self.X_list, self.y_list = zip(*c)

        for e in range(epochs):
            for i in range(len(self)):
                X, y = self.__getitem__(i)
                yield X, y

            self.on_epoch_end()

    def __getitem__(self, index):

        X_batch = self.X_list[index * self.batch_size:(index + 1) * self.batch_size]
        y_batch = self.y_list[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(X_batch, y_batch)
        return X, y

    def on_epoch_end(self, shuffle=False):
        if shuffle:
            c = list(zip(self.X_list, self.y_list))
            random.shuffle(c)
            self.X_list, self.y_list = zip(*c)

    def __get_data(self, X_batch, y_batch):
        y = np.asarray(y_batch)
        X = dataaugmentation(np.asarray(X_batch))
        X = sample_normalization(X, '255_to_0_1')

        return X, y

def build_model(pretrain=True, verbose=False):
    encoder_inputs = keras.Input(shape=(512, 512, 3))

    if pretrain:
        densenet121 = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    else:
        densenet121 = applications.DenseNet121(weights=None, include_top=False)
    densenet121.trainable = True

    x = densenet121(encoder_inputs)
    encoder_outputs = layers.GlobalAveragePooling2D(name='latent_z')(x)

    model = keras.Model(encoder_inputs, encoder_outputs, name="densenet-121_feature_and_regression")

    if verbose:
        model.summary()

    return model

def distance_matrix(A):
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1,1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return D

def custom_distance_loss(y_true, y_pred):
    D_x = distance_matrix(y_pred)
    D_x = D_x / tf.reduce_max(D_x)
    D_y = distance_matrix(y_true)
    D_y = D_y / tf.reduce_max(D_y)
    distance_loss = tf.keras.losses.MeanAbsoluteError()(D_y, D_x)
    return distance_loss