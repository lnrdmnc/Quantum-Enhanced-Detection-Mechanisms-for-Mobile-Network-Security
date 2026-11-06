"""Test del modulo dati su un CSV FITTIZIO con lo stesso schema del dataset reale.

I valori sono generati qui solo per verificare il codice: non sono dati di ricerca.
"""
import numpy as np
import pandas as pd
import pytest

from qdetect import data as D


def fake_csv(path, n=6000, seed=0, bad_labels=5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "freq1": rng.choice([5180, 5200, 5240, 5745, 5765, 5785, 5805], n),
        # valori interi e pochi livelli → molti duplicati, come nel dataset reale
        "noise": rng.integers(-110, -100, n),
        "max_magnitude": rng.integers(30, 60, n) + 5 * y,
        "total_gain_db": rng.choice([40, 68, 90], n),
        "base_pwr_db": rng.integers(370, 425, n),
        "rssi": rng.integers(0, 20, n) + 10 * y,
        "relpwr_db": rng.integers(5, 17, n),
        "avgpwr_db": rng.integers(45, 60, n),
    })
    df["label"] = np.where(y == 1, "malicious", "benign")
    df["label_encoded"] = y.astype(object)
    df.loc[: bad_labels - 1, "label_encoded"] = "x"  # etichette non valide da scartare
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "train_dataset.csv"
    fake_csv(p)
    return p


def test_sampling_is_independent_of_chunksize(csv_path):
    a, sa = D.sample_per_class(csv_path, cap_per_class=1000, seed=7, chunksize=700)
    b, sb = D.sample_per_class(csv_path, cap_per_class=1000, seed=7, chunksize=5000)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb
    assert sa["rows_dropped_bad_label"] == 5
    assert (a["label_encoded"].value_counts() == 1000).all()
    assert a["row_id"].is_monotonic_increasing and a["row_id"].is_unique


def test_sampling_keeps_original_rows(csv_path):
    raw = pd.read_csv(csv_path)
    s, _ = D.sample_per_class(csv_path, cap_per_class=500, seed=1, chunksize=999)
    orig = raw.iloc[s["row_id"].to_numpy()]
    np.testing.assert_array_equal(orig[D.FEATURES].to_numpy(), s[D.FEATURES].to_numpy())
    np.testing.assert_array_equal(orig["label_encoded"].astype(int).to_numpy(), s["label_encoded"].to_numpy())


def test_sampling_takes_all_when_cap_too_big(csv_path):
    with pytest.warns(UserWarning):
        s, st = D.sample_per_class(csv_path, cap_per_class=10_000, seed=1)
    assert len(s) == sum(st["rows_per_class_in_file"].values())


@pytest.mark.parametrize("strategy", ["group", "block"])
def test_split_disjoint_and_nested(csv_path, strategy):
    df, _ = D.sample_per_class(csv_path, cap_per_class=2500, seed=3)
    sp = D.make_split(df, n_train=500, n_train_large=1500, n_val=300, n_test=600,
                      strategy=strategy, n_blocks=50, seed=11)
    assert set(sp.idx_train) <= set(sp.idx_train_large)
    g = D.feature_groups(sp.X)
    for a, b in [(sp.idx_train_large, sp.idx_val), (sp.idx_train_large, sp.idx_test),
                 (sp.idx_val, sp.idx_test)]:
        assert not set(a) & set(b)
        assert not set(g[a]) & set(g[b])
    assert len(sp.idx_train) == 500 and len(sp.idx_test) == 600
    # stratificazione: proporzione di positivi vicina a quella del pool
    assert abs(sp.y[sp.idx_train].mean() - 0.5) < 0.05


def test_split_is_deterministic(csv_path):
    df, _ = D.sample_per_class(csv_path, cap_per_class=2000, seed=3)
    a = D.make_split(df, n_train=300, n_train_large=900, n_val=200, n_test=400, seed=5)
    b = D.make_split(df, n_train=300, n_train_large=900, n_val=200, n_test=400, seed=5)
    for name in ("idx_train", "idx_train_large", "idx_val", "idx_test"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_save_load_roundtrip(csv_path, tmp_path):
    df, st = D.sample_per_class(csv_path, cap_per_class=2000, seed=3)
    sp = D.make_split(df, n_train=300, n_train_large=900, n_val=200, n_test=400)
    p = D.save_split(sp, tmp_path / "split.npz", extra_meta={"sampling": st})
    sp2 = D.load_split(p)
    np.testing.assert_array_equal(sp.X, sp2.X)
    np.testing.assert_array_equal(sp.idx_test, sp2.idx_test)
    assert sp2.meta["sampling"]["cap_per_class"] == 2000
    assert sp2.features == D.FEATURES


def test_prepare_fits_only_on_train(csv_path):
    df, _ = D.sample_per_class(csv_path, cap_per_class=2000, seed=3)
    sp = D.make_split(df, n_train=400, n_train_large=1200, n_val=200, n_test=400)
    P = D.prepare(sp)
    np.testing.assert_allclose(P.X_train.mean(axis=0), 0, atol=1e-9)
    raw = sp.X[sp.idx_train]
    np.testing.assert_allclose(P.preprocessor.named_steps["scaler"].mean_, raw.mean(axis=0))

    Q = D.prepare(sp, k_best=4, angle_scale=2.0)
    assert Q.X_test.shape == (400, 4) and len(Q.features) == 4
    assert np.all(np.abs(Q.X_test) < np.pi)
    # max_magnitude e rssi dipendono dall'etichetta nel CSV fittizio
    assert {"max_magnitude", "rssi"} <= set(Q.features)


def test_duplicate_report_floor():
    X = np.array([[1, 1], [1, 1], [1, 1], [2, 2], [3, 3]], dtype=float)
    y = np.array([0, 1, 1, 0, 1])
    r = D.duplicate_report(X, y)
    assert r["unique_vectors"] == 3
    assert r["conflicting_vectors"] == 1
    assert r["bayes_error_floor"] == pytest.approx(1 / 5)
