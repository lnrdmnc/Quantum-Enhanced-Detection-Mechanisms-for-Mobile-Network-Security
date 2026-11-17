import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from qdetect import quantum as Q  # noqa: E402


def toy(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 0.6, (n, d))
    X[:, 0] += 2.5 * y  # feature 0 → qubit 0: cos(angolo) diverso tra le classi
    return np.pi * np.tanh(X / 2), y


@pytest.mark.parametrize("readout,optimizer", [("affine", "adam"), ("prob", "adam"), ("prob", "qng_diag")])
def test_training_learns_toy_problem(readout, optimizer):
    X, y = toy()
    Xv, yv = toy(seed=1)
    # QAOA-like: RZ e ZZ commutano con Z0, quindi a parametri piccoli ⟨Z0⟩ ≈ cos(x0)
    # e il problema è risolvibile; serve a verificare che loss, gradienti e
    # early stopping funzionino, non a valutare il modello.
    edges, coup = Q.edges_from_correlation(X)
    c = Q.qaoa_rf_aware(4, 1, edges, coup)
    r = Q.train(c, X, y, Xv, yv, readout=readout, optimizer=optimizer, steps=60, batch=16,
                lr=0.05, eval_every=20, patience=5, seed=0)
    assert r.best_val_auc > 0.8
    s = Q.predict_score(r.params, c, Xv, readout)
    assert s.shape == (len(yv),) and np.all((s >= 0) & (s <= 1))


def test_broadcast_matches_single_sample():
    X, _ = toy(n=5)
    edges, coup = Q.edges_from_correlation(toy(n=200)[0])
    for c in (Q.qaoa_rf_aware(4, 2, edges, coup), Q.vqe_sel(4, 2), Q.qaoa_global(4, 2)):
        w = np.random.default_rng(0).normal(0, 0.3, c.n_params)
        batch = np.asarray(c.fn(w, X))
        single = np.array([float(c.fn(w, x)) for x in X])
        np.testing.assert_allclose(batch, single, atol=1e-10)
    assert Q.n_trainable(Q.vqe_sel(4, 2), "affine") == 26
