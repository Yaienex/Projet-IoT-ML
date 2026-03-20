from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler

def align_labels(true_labels, pred_labels):
    """
    Aligne les labels prédits sur les labels réels via l'assignation
    optimale (algorithme hongrois sur la matrice de co-occurrence).
    Les points bruit (label=-1) sont ignorés dans l'alignement.
    """
    # On ne travaille que sur les points non bruités
    mask = pred_labels != -1
    y_true = true_labels[mask]
    y_pred = pred_labels[mask]

    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)

    # Matrice de co-occurrence
    cost = np.zeros((len(classes_pred), len(classes_true)), dtype=int)
    for i, cp in enumerate(classes_pred):
        for j, ct in enumerate(classes_true):
            cost[i, j] = np.sum((y_pred == cp) & (y_true == ct))

    # Maximisation → on minimise le négatif
    row_ind, col_ind = linear_sum_assignment(-cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[classes_pred[r]] = classes_true[c]

    # Remapping
    y_pred_aligned = np.array([mapping.get(p, -1) for p in pred_labels])
    return y_pred_aligned, mapping


def dbscan_predict(X_train, X_test, train_pred, y_test):
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, idx = nbrs.kneighbors(X_test)
    test_pred = train_pred[idx.flatten()]

    test_pred_aligned, mapping = align_labels(y_test, test_pred)
    mask_eval = (test_pred_aligned != -1) & (y_test != -1)
    y_eval_true = y_test[mask_eval]
    y_eval_pred = test_pred_aligned[mask_eval]

    return y_eval_true, y_eval_pred


def train_test_data(df: DataFrame):
    # data
    X = df.drop(columns=["Type"])
    # label
    Y = df["Type"]
    # We remove any incorrect value
    # Infinity value handling
    toremove = np.isinf(X).any(axis=1)
    X = X[~toremove]
    Y = Y[~toremove]
    # NaN value handling
    toremove = np.isnan(X).any(axis=1)
    X = X[~toremove]
    Y = Y[~toremove]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # we convert the label into numeric value
    le = LabelEncoder()
    y = le.fit_transform(Y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    return X_train, X_test, y_train, y_test, le
