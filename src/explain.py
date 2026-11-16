"""SHAP explanations, always expressed for the *malignant* class."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from .model import TrainedModel


class Explainer:
    """Wraps a SHAP explainer so every model is explained the same way.

    Values are in the model's native output space: probability for Random Forest,
    log-odds for Gradient Boosting and Logistic Regression. They are flipped so a
    positive value always pushes the prediction towards "malignant".
    """

    def __init__(self, model: TrainedModel):
        self.model = model
        background = model.scaler.transform(model.data.X_train)
        clf = model.clf
        if isinstance(clf, LogisticRegression):
            self._explainer = shap.LinearExplainer(clf, shap.maskers.Independent(background, max_samples=len(background)))
            self.space = "log-odds"
        else:
            self._explainer = shap.TreeExplainer(clf)
            self.space = "probability" if clf.__class__.__name__ == "RandomForestClassifier" else "log-odds"

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        scaled = self.model.scaler.transform(X)
        exp = self._explainer(scaled)
        values = np.asarray(exp.values)
        base = np.asarray(exp.base_values)
        if values.ndim == 3:  # (n, features, classes) -> benign class
            values, base = values[:, :, 1], base[:, 1]
        base = np.broadcast_to(base, (len(X),)).astype(float)
        # Flip: explain malignant instead of benign.
        values = -values
        base = (1.0 - base) if self.space == "probability" else -base
        return shap.Explanation(values=values, base_values=base,
                                data=X.to_numpy(), feature_names=list(X.columns))

    def reconstructed(self, exp: shap.Explanation) -> np.ndarray:
        """base + sum(values): what the explanation says the model output is."""
        return exp.base_values + exp.values.sum(axis=1)

    def actual(self, X: pd.DataFrame) -> np.ndarray:
        """What the model really outputs (malignant class), in `self.space`."""
        if self.space == "probability":
            return self.model.malignant_probability(X)
        return -self.model.pipeline.decision_function(X)  # log-odds of malignant

    def max_error(self, X: pd.DataFrame) -> float:
        exp = self.explain(X)
        return float(np.abs(self.reconstructed(exp) - self.actual(X)).max())


def top_features(exp: shap.Explanation, n: int = 8) -> list[str]:
    importance = np.abs(exp.values).mean(axis=0)
    order = np.argsort(importance)[::-1][:n]
    return [exp.feature_names[i] for i in order]