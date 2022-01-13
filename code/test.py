import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import h5py

from .model import custom_distance_loss
from .utils import get_scores_euclidean

TRAIN_LABEL = r'D:\AITEM\Data\ICIAP_2021_challenge\Train.csv'
TEST_FOLDER = r'..\data\Test'

train_black_list = ['Image_0736.png']

## ------------- Load data ---------------- ##

train_df = pd.read_csv(TRAIN_LABEL, header = None, names = ['file', 'covid_pct', 'patient'])

for f in train_black_list:
    train_df = train_df[train_df.file != f]

X_test = []

for file in os.listdir(TEST_FOLDER):
    if not file.endswith('.png'):
        continue

    print(file, end='\r')

    path = os.path.join(TEST_FOLDER, file)
    img = cv2.imread(path)

    img = cv2.resize(img, (512, 512))
    q = np.quantile(img, 0.9)
    img = img / q

    X_test.append(img.astype('float32'))

X_test = np.asarray(X_test)

print('testing', X_test.shape)

## ------------- Feature extraction  ---------------- ##

model = tf.keras.models.load_model((r'..\models\model'),
                                        custom_objects =
                                            {'custom_distance_loss': custom_distance_loss})


test_features_tensor = model.predict(X_test, batch_size = 1)
test_features = {}

for i, f in enumerate(test_features_tensor):
    test_features[os.listdir(TEST_FOLDER)[i]] = f

feature_file = r'..\features\test-features.h5'
print('Saving extracted features in', feature_file)
try:
    hf = h5py.File(feature_file, 'w')
except:
    print('Locked file... I try close and reopen')
    hf.close()
    hf = h5py.File(feature_file, 'w')

for i, f in enumerate(test_features):
    hf.create_dataset(f, data=test_features[f])
hf.close()

train_features = {}
train_features_tensor = []

feature_file = r'..\features\refset-features.h5'
print('Opening extracted reference set features from', feature_file)
try:
    hf = h5py.File(feature_file, 'r')
    for f in list(hf.keys()):
        print(f, end='\r')
        if f in train_black_list:
            print('---skip')
            continue
        train_features[f] = hf.get(f)[()]
        train_features_tensor.append(hf.get(f)[()])
except:
    print('Locked file... I try close and reopen')
    hf.close()
    hf = h5py.File(feature_file, 'r')
hf.close()

train_features_tensor = np.asarray(train_features_tensor)


## ------------- Compute predictions  ---------------- ##

test_pred_df = None

for i, f in enumerate(test_features):
    p, _ = get_scores_euclidean(train_features_tensor, test_features[f], train_df['covid_pct'].values,
                                train_df['file'].values, n_neigh=21)

    pred_e = {}
    pred_e['file'] = [f]
    pred_e['covid_pct'] = [p]

    pred_e_df = pd.DataFrame(pred_e)

    if test_pred_df is None:
        test_pred_df = pred_e_df
    else:
        test_pred_df = pd.concat([test_pred_df, pred_e_df])

test_pred_df.to_csv(r'..\predictions\test_predictions.csv')