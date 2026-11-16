import numpy as np
import pytest

from src.explain import Explainer, top_features
from src.model import MODELS, load_data, train


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.mark.parametrize("name", list(MODELS))
def test_model_is_accurate(name, data):
    m = train(name, data)
    assert m.metrics["Accuracy"] > 0.93
    assert m.metrics["ROC AUC"] > 0.97


@pytest.mark.parametrize("name", list(MODELS))
def test_shap_is_additive(name, data):
    """base + sum(shap) must reproduce the model output."""
    m = train(name, data)
    assert Explainer(m).max_error(data.X_test) < 1e-3


@pytest.mark.parametrize("name", list(MODELS))
def test_explanations_are_for_malignant_class(name, data):
    m = train(name, data)
    ex = Explainer(m)
    exp = ex.explain(data.X_test)
    # Ranking by malignant probability must match ranking by the explained output.
    p = m.malignant_probability(data.X_test)
    from scipy.stats import spearmanr
    corr = spearmanr(p, ex.reconstructed(exp)).statistic
    assert corr > 0.99


def test_top_features_returns_requested_count(data):
    m = train("Gradient Boosting", data)
    exp = Explainer(m).explain(data.X_test)
    assert len(top_features(exp, 5)) == 5