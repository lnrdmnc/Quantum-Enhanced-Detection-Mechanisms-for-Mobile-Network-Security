import numpy as np
import pytest

from qdetect import oneclass as O
from qdetect import qkernel as K


# --- soglia conforme e rischio ------------------------------------------------
def test_conformal_threshold_controls_fpr():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=999)
    h = O.conformal_threshold(cal, 0.01)
    test = rng.normal(size=200_000)
    assert abs((test > h).mean() - 0.01) < 0.004
    assert O.conformal_threshold(cal, 0.0001) == np.inf  # k > n: livello non stimabile


def test_fpr_upper_and_min_negatives():
    assert O.min_negatives_for_zero_fp(0.01) == 299
    assert O.min_negatives_for_zero_fp(0.001) == 2995
    assert O.min_negatives_for_zero_fp(0.0001) == 29956
    assert O.fpr_upper95(0, 299) <= 0.01 < O.fpr_upper95(0, 298)


def test_row_features_dimensions():
    Z = np.arange(12.0).reshape(3, 4)
    assert O.row_features(Z, "first").shape == (3, 4)
    assert O.row_features(Z, "second").shape == (3, 4 + 10)
    assert O.row_features(Z, "third").shape == (3, 4 + 10 + 20)


def test_run_grid_detects_shift():
    rng = np.random.default_rng(1)
    names = ["a", "b", "c"]

    def data(n):
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, 3))
        X[:, 1] += 3 * y
        return X, y

    Xtr, ytr = data(2000)
    Xc, yc = data(2000)
    Xt, yt = data(2000)
    res = O.run_grid(Xtr, ytr, Xc, yc, {"test": (Xt, yt, (Xt[:, 0] > 0).astype(float))}, names,
                     feature_sets={"ab": ["a", "b"]}, kinds=("first",), verbose=False)
    r = res[(res.stratum == "all") & (res.alpha == 0.01) & (res.model == "mahalanobis")].iloc[0]
    assert r.auroc > 0.9 and r.fpr < 0.03
    assert set(res.training) == {"benign_only", "supervised"}


# --- kernel quantistico --------------------------------------------------------
def test_states_normalized_and_match_reference():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (5, 4))
    S = K.states(X)
    np.testing.assert_allclose((S ** 2).sum(1), 1, atol=1e-12)
    # senza entanglement lo stato è un prodotto tensoriale di stati a 1 qubit
    Ss = K.states(X, entangled=False)
    for n in range(5):
        prod = np.array([1.0])
        for j in reversed(range(4)):  # qubit j = bit j dell'indice
            c, s = np.cos(X[n, j] / 2), np.sin(X[n, j] / 2)
            t = 0.7 * X[n, j]
            c2, s2 = np.cos(t / 2), np.sin(t / 2)
            q = np.array([c2 * c - s2 * s, s2 * c + c2 * s])
            prod = np.kron(prod, q)
        np.testing.assert_allclose(Ss[n], prod, atol=1e-12)


def test_cnot_ring_matches_pennylane_if_available():
    qml = pytest.importorskip("pennylane")
    x = np.array([0.3, -1.2, 2.0, 0.7])
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circ():
        for j in range(4):
            qml.RY(x[j], wires=j)
        for j in range(4):
            qml.CNOT(wires=[j, (j + 1) % 4])
        for j in range(4):
            qml.RY(0.7 * x[j], wires=j)
        return qml.state()

    ref = np.real(circ())
    # PennyLane usa il qubit 0 come bit più significativo: riordino gli indici
    ours = K.states(x[None])[0]
    perm = [int(format(i, "04b")[::-1], 2) for i in range(16)]
    np.testing.assert_allclose(ours[perm], ref, atol=1e-10)


def test_kernel_and_geometry():
    rng = np.random.default_rng(2)
    X = rng.uniform(-np.pi, np.pi, (30, 5))
    Kq = K.fidelity_kernel(X)
    np.testing.assert_allclose(np.diag(Kq), 1, atol=1e-12)
    assert np.allclose(Kq, Kq.T)
    assert K.geometric_difference(Kq, Kq) == pytest.approx(1.0, rel=1e-6)
    assert K.geometric_difference(Kq, np.eye(30)) >= 1.0
    assert 1 <= K.effective_rank(Kq) <= 30
    assert K.hoeffding_shots(100, 0.05) > K.hoeffding_shots(10, 0.05)
