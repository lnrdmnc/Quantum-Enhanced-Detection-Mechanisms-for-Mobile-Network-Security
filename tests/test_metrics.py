import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qdetect import metrics as M


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.8, len(y))
    weak = y + rng.normal(0, 2.0, len(y))
    return y, good, weak


def test_delong_auc_matches_sklearn(scores):
    y, good, _ = scores
    auc, lo, hi = M.auroc_ci(y, good)
    assert auc == pytest.approx(roc_auc_score(y, good), abs=1e-12)
    assert lo < auc < hi


def test_delong_ci_close_to_bootstrap(scores):
    y, good, _ = scores
    _, lo, hi = M.auroc_ci(y, good)
    _, blo, bhi = M.bootstrap_auroc_ci(y, good, n_boot=1000, seed=1)
    assert abs(lo - blo) < 0.01 and abs(hi - bhi) < 0.01


def test_delong_test(scores):
    y, good, weak = scores
    r = M.delong_test(y, good, weak)
    assert r["diff"] > 0 and r["p_value"] < 1e-6
    same = M.delong_test(y, good, good)
    assert same["p_value"] == 1.0


def test_delong_handles_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.9])
    auc, _, _ = M.auroc_ci(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s))


def test_pd_at_pfa():
    y = np.array([0] * 100 + [1] * 100)
    s = np.r_[np.linspace(0, 1, 100), np.linspace(0.5, 1.5, 100)]
    assert M.pd_at_pfa(y, s, 0.0) == pytest.approx(0.51, abs=0.02)
    assert M.pd_at_pfa(y, s, 1.0) == 1.0


def test_orient_scores(scores):
    y, good, _ = scores
    assert M.orient_scores(y, good) == 1
    assert M.orient_scores(y, -good) == -1


def test_evaluate_and_append(scores, tmp_path):
    y, good, _ = scores
    p = 1 / (1 + np.exp(-(good - 0.5) * 3))
    row = M.evaluate("toy", y, p, family="classical", seed=1, n_train=10, n_params=2)
    assert list(row) == M.RESULT_COLUMNS
    assert 0 < row["fp"] < len(y)
    out = tmp_path / "res.csv"
    M.append_results(out, [row])
    M.append_results(out, [row])
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("model,family")
    with pytest.raises(ValueError):
        M.evaluate("bad", y, np.full(len(y), np.nan), family="x")
