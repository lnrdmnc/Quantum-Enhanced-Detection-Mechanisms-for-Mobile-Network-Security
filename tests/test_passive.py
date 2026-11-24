import numpy as np
import pandas as pd

from qdetect import passive as P

CH = [5180, 5200, 5745, 5765]


def sweep(rng, jam=None, power=0.0, rows_per_ch=150):
    """File passive fittizio: scansione sequenziale dei canali; il jammer alza rssi
    solo sul suo canale."""
    parts = []
    for c in CH:
        d = pd.DataFrame({k: rng.normal(10, 1, rows_per_ch) for k in P.ALL7})
        d.insert(0, "freq1", c)
        if c == jam:
            d["rssi"] += 3 + power
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_channel_windows():
    rng = np.random.default_rng(0)
    d = sweep(rng, jam=5745, power=2)
    recs = P.channel_windows(d, windows=(8, 128, 200))
    assert {r["W"] for r in recs} == {8, 128}  # 200 > righe disponibili per canale
    r = next(x for x in recs if x["channel"] == 5745 and x["W"] == 128)
    assert r["m_rssi"] > 13 and r["n_rows_channel"] == 150
    assert len(P.feature_columns(P.CORE, "second")) == 5 + 15


def test_file_split_deterministic():
    a = P.file_split([f"f{i}.csv" for i in range(1000)])
    b = P.file_split([f"f{i}.csv" for i in range(1000)])
    assert (a == b).all() and abs((a == "train").mean() - 0.5) < 0.06


def test_channel_matrix_diagonal():
    rng = np.random.default_rng(1)
    recs = []
    for i in range(400):
        lab, jam = ("benign", None) if i < 240 else ("malicious", [5180, 5745][i % 2])
        for r in P.channel_windows(sweep(rng, jam, power=3), windows=(32,)):
            recs.append({**r, "file_path": f"f{i}", "label": lab, "benign_subtype": "floor" if lab == "benign" else "",
                         "band": "5ghz", "channel_mhz": jam if jam else np.nan, "power_dbm": 3.0 if jam else np.nan})
    F = pd.DataFrame(recs)
    F["split"] = F.file_path.map(dict(zip(F.file_path.unique(), P.file_split(F.file_path.unique()))))
    M = P.channel_matrix(F, "5ghz", 32, P.ALL7, models=["mahalanobis"])
    diag = M[M.monitored_mhz == M.jammer_mhz].auroc
    off = M[(M.monitored_mhz != M.jammer_mhz)].auroc
    assert (diag > 0.95).all()
    assert (off < 0.75).all()  # nessuna scorciatoia nei dati fittizi


def test_channel_windows_multiple_and_filter():
    rng = np.random.default_rng(2)
    d = sweep(rng, jam=5745, power=2, rows_per_ch=100)
    recs = P.channel_windows(d, windows=(8, 32), max_windows=4, channels=[5745])
    assert {r["channel"] for r in recs} == {5745}
    assert sum(r["W"] == 8 for r in recs) == 4 and sum(r["W"] == 32 for r in recs) == 3
    assert sorted(r["window"] for r in recs if r["W"] == 32) == [0, 1, 2]
