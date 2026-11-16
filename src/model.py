"""Data loading, model training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

MODELS = {
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
}


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def load_data(test_size: float = 0.25) -> Dataset:
    """Wisconsin breast-cancer data. In sklearn, target 1 = benign, 0 = malignant."""
    raw = load_breast_cancer(as_frame=True)
    X, y = raw.data, raw.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    return Dataset(X_tr, X_te, y_tr, y_te)


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    data: Dataset
    metrics: dict
    confusion: np.ndarray

    @property
    def scaler(self) -> StandardScaler:
        return self.pipeline.named_steps["scaler"]

    @property
    def clf(self):
        return self.pipeline.named_steps["clf"]

    def malignant_probability(self, X: pd.DataFrame) -> np.ndarray:
        """P(malignant) = 1 - P(benign); class 1 is benign in this dataset."""
        return 1.0 - self.pipeline.predict_proba(X)[:, 1]


def train(name: str, data: Dataset | None = None) -> TrainedModel:
    data = data or load_data()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", MODELS[name]())])
    pipe.fit(data.X_train, data.y_train)

    proba_malignant = 1.0 - pipe.predict_proba(data.X_test)[:, 1]
    y_true_malignant = (data.y_test == 0).astype(int).to_numpy()
    y_pred_malignant = (proba_malignant >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true_malignant, y_pred_malignant),
        "Precision": precision_score(y_true_malignant, y_pred_malignant),
        "Recall": recall_score(y_true_malignant, y_pred_malignant),
        "F1": f1_score(y_true_malignant, y_pred_malignant),
        "ROC AUC": roc_auc_score(y_true_malignant, proba_malignant),
    }
    cm = confusion_matrix(y_true_malignant, y_pred_malignant)
    return TrainedModel(name, pipe, data, metrics, cm)