"""Rilevamento per canale sul passive scan (dataset per file).

Ogni file passive è una scansione sequenziale dei canali della banda
(5 GHz: 5180 → 5825, 13 canali; 2,4 GHz: 2412 → 2462, 11 canali), circa 4.700 righe.
Il prefisso di 128 righe usato dal protocollo contiene quindi solo il primo canale
(5180 o 2412 MHz): un jammer su un altro canale non vi compare.

Qui ogni file viene ridotto a **finestre per canale**: per ogni canale c, le prime W
righe con ``freq1 == c`` (nell'ordine del file). Per ogni (file, canale, W) si salvano
media e covarianza delle feature. Gli esperimenti usano un rilevatore che monitora un
canale c ed è addestrato sui soli benigni a quel canale. La **matrice canale
monitorato × canale del jammer** separa il rilevamento vero (diagonale e canali
adiacenti) dalle scorciatoie: se un jammer su j è "rilevato" su un canale lontano da
j, il modello sta usando differenze di sessione o acquisizione, non il jamming.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Optional

import numpy as np
import pandas as pd

CORE = ["max_magnitude", "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db"]  # senza noise e gain
ALL7 = ["noise", "max_magnitude", "total_gain_db", "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db"]
WINDOWS = (8, 32, 128)


def channel_windows(df: pd.DataFrame, fields: list[str] = ALL7, windows: Iterable[int] = WINDOWS,
                    max_windows: int = 1, channels: Optional[Iterable[int]] = None) -> list[dict]:
    """Per ogni canale del file e ogni W: media e covarianza (triangolo superiore) di
    finestre di W righe consecutive a quel canale (fino a ``max_windows`` finestre non
    sovrapposte, ``window`` = indice). ``channels`` limita i canali considerati."""
    f1 = df["freq1"].to_numpy()
    X = df[fields].to_numpy(dtype=float)
    keep = set(int(c) for c in channels) if channels is not None else None
    out = []
    iu = np.triu_indices(len(fields))
    for c in pd.unique(f1):
        if keep is not None and int(c) not in keep:
            continue
        rows = X[f1 == c]
        rows = rows[np.isfinite(rows).all(1)]
        for w in windows:
            for k in range(min(max_windows, len(rows) // w)):
                a = rows[k * w:(k + 1) * w]
                mu = a.mean(0)
                cov = np.cov(a, rowvar=False, bias=True)[iu] if w > 1 else np.zeros(len(iu[0]))
                out.append({"channel": int(c), "W": int(w), "window": k, "n_rows_channel": int(len(rows)),
                            **{f"m_{k2}": v for k2, v in zip(fields, mu)},
                            **{f"c_{fields[i]}_{fields[j]}": v for i, j, v in zip(iu[0], iu[1], cov)}})
    return out


def feature_columns(fields: list[str], kind: str) -> list[str]:
    """``mean``: medie; ``second``: medie + covarianze."""
    cols = [f"m_{k}" for k in fields]
    if kind == "second":
        cols += [f"c_{a}_{b}" for a, b in itertools.combinations_with_replacement(fields, 2)]
    elif kind != "mean":
        raise ValueError(kind)
    return cols


def file_split(paths: Iterable[str], seed: str = "split-v1") -> np.ndarray:
    """Split per file deterministico 50/25/25 (train/cal/test) da hash del percorso."""
    import hashlib
    u = np.array([int(hashlib.sha256((seed + p).encode()).hexdigest()[:12], 16) / 16 ** 12 for p in paths])
    return np.where(u < 0.5, "train", np.where(u < 0.75, "cal", "test"))


def channel_matrix(
    F: pd.DataFrame, band: str, W: int, fields: list[str], kind: str = "mean",
    benign_subtype: str = "floor", alphas=(0.01, 0.001), seed: int = 42,
    models: Optional[list[str]] = None, test_benign_subtype: Optional[str] = None,
) -> pd.DataFrame:
    """Per ogni canale monitorato c della banda: modelli one-class addestrati sulle finestre
    benigne (train) a c, soglia conforme su quelle di calibrazione, valutati sul test
    contro i file malevoli di ogni canale del jammer j (finestre al canale c).

    ``F`` ha una riga per (file, canale, W) con colonne del manifest + feature + ``split``.
    """
    from .oneclass import conformal_threshold, fpr_upper95, one_class_models, supervised_reference
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    cols = feature_columns(fields, kind)
    G = F[(F.band == band) & (F.W == W)]
    jam_channels = sorted(G[G.label == "malicious"].channel_mhz.dropna().unique())
    rows = []
    for c in sorted(G.channel.unique()):
        H = G[G.channel == c]
        ben = H[(H.label == "benign") & (H.benign_subtype == benign_subtype)]
        tr_b, cal_b, te_b = (ben[ben.split == s] for s in ("train", "cal", "test"))
        if test_benign_subtype is not None:  # es. calibrazione in camera, test in ambiente reale
            other = H[(H.label == "benign") & (H.benign_subtype == test_benign_subtype)]
            te_b = other[other.split == "test"]
        if len(tr_b) < 20 or len(cal_b) < 20 or len(te_b) < 20:
            continue
        sc = StandardScaler().fit(tr_b[cols])
        sc.scale_ = np.maximum(sc.scale_, 1e-6)
        tf = lambda d: np.clip(sc.transform(d[cols]), -1e6, 1e6)  # noqa: E731
        gens = list(one_class_models(tf(tr_b), seed=seed))
        mal_tr = H[(H.label == "malicious") & (H.split == "train")]
        if len(mal_tr):
            Xs = np.r_[tf(tr_b), tf(mal_tr)]
            ys = np.r_[np.zeros(len(tr_b)), np.ones(len(mal_tr))]
            gens.append(supervised_reference(Xs, ys, seed))
        if models is not None:
            gens = [g for g in gens if g[0] in models]
        for name, fn in gens:
            s_cal = fn(tf(cal_b))
            s_neg = fn(tf(te_b))
            for j in jam_channels:
                for p, mal in H[(H.label == "malicious") & (H.split == "test") & (H.channel_mhz == j)].groupby("power_dbm"):
                    s_pos = fn(tf(mal))
                    y = np.r_[np.zeros(len(s_neg)), np.ones(len(s_pos))]
                    s = np.r_[s_neg, s_pos]
                    r = {"band": band, "W": W, "kind": kind, "fields": "core" if fields == CORE else "all7",
                         "benign_subtype": benign_subtype, "test_benign": test_benign_subtype or benign_subtype,
                         "monitored_mhz": int(c), "jammer_mhz": int(j),
                         "power_dbm": p, "model": name,
                         "training": "supervised" if name == "supervised_rf" else "benign_only",
                         "n_train_benign": len(tr_b), "n_cal": len(cal_b), "n_neg": len(s_neg), "n_pos": len(s_pos),
                         "auroc": roc_auc_score(y, s)}
                    for a in alphas:
                        h = conformal_threshold(s_cal, a)
                        fp = int((s_neg > h).sum())
                        r[f"tpr@{a:g}"] = float((s_pos > h).mean())
                        r[f"fpr@{a:g}"] = fp / len(s_neg)
                        r[f"fpr_up95@{a:g}"] = fpr_upper95(fp, len(s_neg))
                        r[f"finite@{a:g}"] = bool(np.isfinite(h))
                    rows.append(r)
    return pd.DataFrame(rows)
