import pandas as pd
import numpy as np
import cv2
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import h5py

from .model import build_model, custom_distance_loss, DataGenerator

TRAIN_FOLDER = r'..\data\Train'
TRAIN_LABEL = r'..\data\Train.csv'

train_black_list = ['Image_0736.png']

## ------------- Load data ---------------- ##

train_df = pd.read_csv(TRAIN_LABEL, header = None, names = ['file', 'covid_pct', 'patient'])

for f in train_black_list:
    train_df = train_df[train_df.file != f]

y_train_all = train_df['covid_pct'].values

X_train = []
y_train = []

i = 0
for file in os.listdir(TRAIN_FOLDER):
    if not file.endswith('.png'):
        continue

    if file in train_black_list:
        continue

    path = os.path.join(TRAIN_FOLDER, file)
    img = cv2.imread(path)
    X_train.append(img)
    y_train.append(y_train_all[i])
    i = i + 1

y_train = np.asarray(y_train)
X_train = np.asarray(X_train)

print('training', X_train.shape, y_train.shape)

## ------------- Training model ---------------- ##

model = build_model(pretrain = True, verbose = True)
model.compile(optimizer=keras.optimizers.SGD(learning_rate = 5e-2), loss = custom_distance_loss)

print('Training')
epochs = 50
batch_size = 12

history = {}
history['loss'] = []

train_datagen = DataGenerator()

for e in range(epochs):

    lossEpochTrain = 0

    batchCnt = 0
    for X_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=batch_size):
        print(batchCnt, '/', int(len(X_train) / batch_size), end='\r')
        batch_history = model.train_on_batch(X_batch, y_batch)
        print(batchCnt, '/', int(len(X_train) / batch_size), 'LOSS:', np.round(batch_history, 5), end='\r')
        lossEpochTrain += batch_history

        batchCnt += 1

        if batchCnt >= len(X_train) / batch_size:
            break

    lossEpochTrain /= batchCnt

    print('Epoch', e)
    print('Train: Loss {:.5f}'.format(lossEpochTrain))
    history['loss'].append(lossEpochTrain)

plt.figure()
plt.plot(history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Distance matrix loss')
plt.show()

model.save(r'..\models\model')

## ------------- Feature extraction from training set ---------------- ##

train_features_tensor = model.predict(X_train/255, batch_size = 1)

train_features = {}

for i, f in enumerate(train_features_tensor):
    train_features[train_df['file'].values[i]] = f

feature_file = r'..\features\refset-features.h5'
print('Saving extracted features in', feature_file)
try:
    hf = h5py.File(feature_file, 'w')
except:
    print('Locked file... I try close and reopen')
    hf.close()
    hf = h5py.File(feature_file, 'w')

for i, f in enumerate(train_features):
    hf.create_dataset(f, data=train_features[f])
hf.close()