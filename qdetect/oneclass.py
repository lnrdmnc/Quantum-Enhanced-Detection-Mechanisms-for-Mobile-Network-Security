"""Rilevamento one-class a rischio fissato (training sui soli benigni).

Segue il protocollo della specifica del relatore, adattato al CSV a righe:

* i modelli vedono solo righe **benigne** di training (Mahalanobis con shrinkage,
  residuo PCA, Isolation Forest, One-Class SVM, Mahalanobis su random Fourier
  features); un Random Forest supervisionato è solo un riferimento separato;
* la soglia è **conforme**: calcolata sui punteggi di righe benigne di calibrazione,
  k = ⌈(n+1)(1−α)⌉; se k > n il livello α non è stimabile con soglia finita;
* sul test si riportano FPR osservato, limite superiore al 95% (Clopper–Pearson
  unilaterale, assumendo righe indipendenti: ipotesi **non verificata**, perché il CSV
  non contiene sessioni né file) e TPR, anche per canale (``freq1``).

Unità di rischio = riga, non file: è un limite dichiarato del CSV.
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ALPHAS = (0.01, 0.001, 0.0001)

# Insiemi di feature (ablation E3). freq1 è il canale: nella specifica è una variabile
# di stratificazione, non un input; "all8" serve solo a confrontarsi con i risultati
# precedenti, che la usavano come feature.
FIELDS7 = ["noise", "max_magnitude", "total_gain_db", "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db"]
FEATURE_SETS = {
    "all8": ["freq1"] + FIELDS7,
    "all7": FIELDS7,
    "core5": [f for f in FIELDS7 if f not in ("noise", "total_gain_db")],
}


# ---------------------------------------------------------------------------
# Rappresentazioni per riga
# ---------------------------------------------------------------------------
def row_features(Z: np.ndarray, kind: str) -> np.ndarray:
    """``first``: le feature standardizzate; ``second``: + prodotti z_i z_j (i ≤ j);
    ``third``: + prodotti z_i z_j z_k (i ≤ j ≤ k).

    Sono l'analogo per singola riga dei momenti della specifica: senza file né ordine
    temporale non esistono finestre su cui calcolare momenti o autocorrelazioni.
    """
    Z = np.asarray(Z, dtype=float)
    p = Z.shape[1]
    out = [Z]
    if kind in ("second", "third"):
        out.append(np.stack([Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(p), 2)], 1))
    if kind == "third":
        out.append(np.stack([Z[:, i] * Z[:, j] * Z[:, k]
                             for i, j, k in itertools.combinations_with_replacement(range(p), 3)], 1))
    if kind not in ("first", "second", "third"):
        raise ValueError(kind)
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# Modelli
# ---------------------------------------------------------------------------
def one_class_models(X_benign: np.ndarray, seed: int = 42, ocsvm_cap: int = 1500
                     ) -> Iterable[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Genera (nome, funzione punteggio) addestrati sulle sole righe benigne.
    Punteggio alto = più anomalo (= più probabilmente malevolo)."""
    B = np.asarray(X_benign)
    lw = LedoitWolf().fit(B)
    yield "mahalanobis", lw.mahalanobis
    pca = PCA(n_components=max(1, min(B.shape[1] - 1, len(B) - 1, 8)), random_state=seed).fit(B)
    yield "pca_residual", lambda x: ((x - pca.inverse_transform(pca.transform(x))) ** 2).sum(1)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(B)
    yield "isolation_forest", lambda x: -iso.score_samples(x)
    bb = B[np.random.default_rng(seed).choice(len(B), min(ocsvm_cap, len(B)), replace=False)]
    oc = OneClassSVM(gamma="scale", nu=0.05).fit(bb)
    yield "ocsvm_rbf", lambda x: -oc.decision_function(x).ravel()
    rff = RBFSampler(gamma=1 / max(1, B.shape[1]), n_components=128, random_state=seed).fit(B)
    cov = LedoitWolf().fit(rff.transform(B))
    yield "rff_mahalanobis", lambda x: cov.mahalanobis(rff.transform(x))


def supervised_reference(X: np.ndarray, y: np.ndarray, seed: int = 42):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                random_state=seed, class_weight="balanced").fit(X, y)
    return "supervised_rf", lambda x: rf.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# Soglia conforme e metriche a rischio fissato
# ---------------------------------------------------------------------------
def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Soglia h tale che P(score > h | benigno) ≤ α, se i punteggi sono scambiabili."""
    s = np.sort(np.asarray(cal_scores))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if 0 < k <= len(s) else np.inf


def fpr_upper95(fp: int, n: int) -> float:
    """Limite superiore unilaterale al 95% (Clopper–Pearson) sul tasso di falsi allarmi."""
    if n == 0:
        return np.nan
    return 1.0 if fp >= n else float(beta.ppf(0.95, fp + 1, n - fp))


def min_negatives_for_zero_fp(alpha: float, conf: float = 0.95) -> int:
    """Negativi indipendenti necessari perché, con zero falsi allarmi, il limite
    superiore sia ≤ α: n ≥ log(1−conf)/log(1−α)."""
    return int(np.ceil(np.log(1 - conf) / np.log(1 - alpha)))


def risk_metrics(scores: np.ndarray, y: np.ndarray, cal_scores: np.ndarray,
                 strata: Optional[np.ndarray] = None, alphas=ALPHAS) -> list[dict]:
    """Per ogni α: soglia conforme, FPR (con limite al 95%) e TPR sul test,
    complessivi e per strato dei positivi (es. canale)."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    neg, pos = y == 0, y == 1
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    rows = []
    groups = [("all", pos)]
    if strata is not None:
        for v in np.unique(strata[pos]):
            groups.append((f"freq1={v:g}", pos & (strata == v)))
    for alpha in alphas:
        h = conformal_threshold(cal_scores, alpha)
        pred = scores > h
        fp, nn = int((pred & neg).sum()), int(neg.sum())
        for name, mask in groups:
            sub = neg | mask
            rows.append({
                "alpha": alpha, "stratum": name, "threshold": h, "finite_threshold": bool(np.isfinite(h)),
                "n_cal": len(cal_scores), "n_neg": nn, "n_pos": int(mask.sum()),
                "fpr": fp / nn, "fpr_upper95": fpr_upper95(fp, nn), "fpr_certified": fpr_upper95(fp, nn) <= alpha,
                "tpr": float(pred[mask].mean()) if mask.any() else np.nan,
                "auroc": auc if name == "all" else roc_auc_score(y[sub], scores[sub]),
                "ap": ap if name == "all" else average_precision_score(y[sub], scores[sub]),
            })
    return rows


def run_grid(
    X_train_raw: np.ndarray, y_train: np.ndarray,
    X_cal_raw: np.ndarray, y_cal: np.ndarray,
    test_sets: dict[str, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    feature_names: list[str],
    feature_sets: dict[str, list[str]] = FEATURE_SETS,
    kinds: Iterable[str] = ("first", "second", "third"),
    seed: int = 42, include_supervised: bool = True, verbose: bool = True,
) -> pd.DataFrame:
    """Griglia insiemi di feature × rappresentazione × modello.

    Scaling stimato solo sulle righe benigne di training; calibrazione sulle sole
    righe benigne di ``X_cal_raw``; ``test_sets``: {nome: (X, y, strati)}.
    """
    y_train, y_cal = np.asarray(y_train).astype(int), np.asarray(y_cal).astype(int)
    rows = []
    for fs_name, cols in feature_sets.items():
        idx = [feature_names.index(c) for c in cols]
        sc = StandardScaler().fit(X_train_raw[y_train == 0][:, idx])
        tf = lambda X: np.clip(sc.transform(X[:, idx]), -10, 10)  # noqa: E731
        for kind in kinds:
            Ftr = row_features(tf(X_train_raw), kind)
            fs = StandardScaler().fit(Ftr[y_train == 0])
            Ftr = fs.transform(Ftr)
            Fcal = fs.transform(row_features(tf(X_cal_raw), kind))[y_cal == 0]
            Fte = {k: fs.transform(row_features(tf(X), kind)) for k, (X, _, _) in test_sets.items()}
            gens = list(one_class_models(Ftr[y_train == 0], seed=seed))
            if include_supervised:
                gens.append(supervised_reference(Ftr, y_train, seed))
            for name, fn in gens:
                t0 = time.perf_counter()
                cal = fn(Fcal)
                for tname, (_, yt, strata) in test_sets.items():
                    s = fn(Fte[tname])
                    for r in risk_metrics(s, yt, cal, strata):
                        rows.append({"feature_set": fs_name, "n_features": len(cols), "representation": kind,
                                     "dim": Ftr.shape[1], "model": name,
                                     "training": "supervised" if name == "supervised_rf" else "benign_only",
                                     "test": tname, **r})
                if verbose:
                    last = [r for r in rows[-200:] if r["model"] == name and r["feature_set"] == fs_name
                            and r["representation"] == kind and r["stratum"] == "all" and r["alpha"] == 0.01]
                    print(f"{fs_name:6s} {kind:6s} {name:17s} " + "  ".join(
                        f"{r['test']}: AUROC={r['auroc']:.3f} TPR@1%={r['tpr']:.3f}" for r in last)
                          + f"  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)
