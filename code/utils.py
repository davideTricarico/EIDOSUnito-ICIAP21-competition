import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

def sample_normalization(X, method='norm'):
    if method == 'norm':
        means = np.mean(X, axis=(1, 2, 3))
        stds = np.std(X, axis=(1, 2, 3))

        X = X - means.reshape((-1, 1, 1, 1))
        X = X / stds.reshape((-1, 1, 1, 1))
    elif method == '255_to_0_1':
        X = X / 255
    elif method == '90th_pct':
        X = X / np.quantile(X, 0.9)
        X = np.where(X > 1, 1, X)
    elif method == 'min_max':
        offset = np.min(X, axis=(1, 2, 3))
        X = X - offset.reshape((-1, 1, 1, 1))
        scale = np.max(X, axis=(1, 2, 3))
        X = X / scale.reshape((-1, 1, 1, 1))

    return X

def get_scores_euclidean(ref_set_mat, query, ref_set_label, ref_patients, n_neigh=10):
    query = query.reshape(1, -1)
    esims = euclidean_distances(ref_set_mat, query)
    esims = esims ** 2
    max_esim = np.max(esims)

    esims = esims / max_esim
    esims = esims.reshape(-1)
    rank = np.argsort(esims).tolist()

    score = 0

    n = min(n_neigh, len(rank))

    i = 0
    l = 0
    neighboors = []
    while i < n:
        if l >= len(rank):
            break
        try:
            f = ref_set_label[rank[l]]
            patient = ref_patients[rank[l]]

            # sim = 100 * abs(esims[rank[l]]) + f
            sim = f
            score += sim

            neighboors.append((f, patient, sim, score / (l + 1)))
        except:
            l += 1
            continue

        i += 1
        l += 1

    return score / n, neighboors

def compute_predictions_leave_one_patient_out(df, features_dict, features_tensor, data_aug_ratio=1, n_neigh=10):
    data_aug_ratio = int(data_aug_ratio)
    pred_df = None
    for i, f in enumerate(features_dict):

        patient = df['patient'].values[i]
        lopo_mask = (df['patient'].values != patient)
        ref_labels = df['covid_pct'].values[lopo_mask]
        ref_patients = df['file'].values[lopo_mask]

        if data_aug_ratio > 1:
            lopo_mask = np.repeat(lopo_mask, data_aug_ratio, axis=0).reshape((-1))
            ref_labels = np.repeat(ref_labels, data_aug_ratio, axis=0).reshape((-1))

        ref_mat = features_tensor[lopo_mask, :]

        pe, neighs_e = get_scores_euclidean(ref_mat, features_dict[f], ref_labels, ref_patients, n_neigh=n_neigh)

        pred_e = {}
        pred_e['file'] = [f]
        pred_e['covid_pct_euclidean'] = [pe]
        pred_e['covid_pct_real'] = [df['covid_pct'].values[i]]
        pred_e['similars_euclidean'] = [neighs_e]

        pred_e_df = pd.DataFrame(pred_e)

        if pred_df is None:
            pred_df = pred_e_df
        else:
            pred_df = pd.concat([pred_df, pred_e_df])
    return pred_df