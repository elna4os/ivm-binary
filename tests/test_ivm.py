import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ivm import IVMBinary


def load_split_breast_cancer(
    test_size: int,
    random_state: int = 42
) -> Tuple[np.ndarray, ...]:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


MODEL_DIR = Path('tests/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def test_training():
    X_train, _, y_train, _ = load_split_breast_cancer(test_size=100)

    clf = IVMBinary()
    clf.fit(X_train, y_train)

    model_dir = Path('tests/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / 'test_breast_cancer_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)


def test_predict_proba():
    _, X_test, _, y_test = load_split_breast_cancer(test_size=100)
    with open(MODEL_DIR / 'test_breast_cancer_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_probas = clf.predict_proba(X_test)
    test_rocauc = roc_auc_score(y_true=y_test, y_score=test_probas)
    print(f'Test ROC-AUC: {test_rocauc}')


def test_predict():
    _, X_test, _, y_test = load_split_breast_cancer(test_size=100)
    with open(MODEL_DIR / 'test_breast_cancer_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_preds = clf.predict(X_test)
    print(classification_report(
        y_true=y_test,
        y_pred=test_preds
    ))
