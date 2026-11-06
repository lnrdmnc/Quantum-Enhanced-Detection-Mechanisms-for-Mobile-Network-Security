"""Metriche comuni e formato unico dei risultati.

Ogni modello produce un *punteggio* continuo sul test (probabilità, energia, margine):
convenzione unica **punteggio più alto = più malevolo**. Se un modello produce
energie con la convenzione opposta, il segno va fissato sul training/validation
(``orient_scores``), mai guardando il test.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

PFA_LEVELS = (0.01, 0.05, 0.10)

RESULT_COLUMNS = [
    "model", "family", "seed", "split", "n_train", "n_params",
    "auroc", "auroc_lo", "auroc_hi", "average_precision",
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "fp", "fn", "threshold",
    *[f"pd_at_pfa_{p:g}" for p in PFA_LEVELS],
    "train_seconds", "infer_seconds", "notes",
]


# ---------------------------------------------------------------------------
# DeLong (Sun & Xu, 2014): varianza dell'AUROC e confronto tra modelli accoppiati
# ---------------------------------------------------------------------------
def _delong_components(y: np.ndarray, scores: np.ndarray):
    """``scores``: (k, n). Restituisce AUROC (k,) e matrice di covarianza (k, k)."""
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("Servono almeno 2 positivi e 2 negativi")
    tx = np.vstack([rankdata(r) for r in pos])
    ty = np.vstack([rankdata(r) for r in neg])
    tz = np.vstack([rankdata(r) for r in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # componenti strutturali dei positivi
    v10 = 1.0 - (tz[:, m:] - ty) / m    # componenti strutturali dei negativi
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, cov


def auroc_ci(y, score, alpha: float = 0.05) -> tuple[float, float, float]:
    """AUROC con intervallo di confidenza di DeLong (asintotico, troncato a [0,1])."""
    auc, cov = _delong_components(y, score)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = norm.ppf(1 - alpha / 2)
    a = float(auc[0])
    return a, float(max(0.0, a - z * se)), float(min(1.0, a + z * se))


def delong_test(y, score_a, score_b) -> dict:
    """Test di DeLong a due code per AUROC(a) − AUROC(b) sullo stesso test set."""
    auc, cov = _delong_components(y, np.vstack([score_a, score_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(auc[0] - auc[1])
    if var <= 0:
        return {"auroc_a": float(auc[0]), "auroc_b": float(auc[1]), "diff": diff, "z": 0.0, "p_value": 1.0}
    z = diff / np.sqrt(var)
    return {"auroc_a": float(auc[0]), "auroc_b": float(auc[1]), "diff": diff,
            "z": float(z), "p_value": float(2 * norm.sf(abs(z)))}


def bootstrap_auroc_ci(y, score, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Intervallo percentile bootstrap (controllo dell'approssimazione di DeLong)."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(roc_auc_score(y, score)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Rilevamento a falso allarme fissato
# ---------------------------------------------------------------------------
def pd_at_pfa(y, score, pfa: float) -> float:
    """Probabilità di rilevamento massima con falso allarme ≤ ``pfa`` (dalla curva ROC)."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= pfa + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def orient_scores(y_fit, score_fit) -> int:
    """+1 se il punteggio è già "alto = malevolo" sui dati di fit, altrimenti −1.

    Da chiamare su train o validation; il segno trovato si applica poi al test.
    """
    return 1 if roc_auc_score(y_fit, score_fit) >= 0.5 else -1


# ---------------------------------------------------------------------------
# Valutazione e salvataggio con schema unico
# ---------------------------------------------------------------------------
def evaluate(
    model: str,
    y_test,
    score,
    *,
    family: str,
    seed: Optional[int] = None,
    split: str = "group",
    n_train: Optional[int] = None,
    n_params: Optional[int] = None,
    threshold: float = 0.5,
    train_seconds: Optional[float] = None,
    infer_seconds: Optional[float] = None,
    notes: str = "",
) -> dict:
    """Riga di risultati. ``score``: alto = malevolo. ``threshold`` va scelta senza test
    (0.5 per probabilità calibrate, oppure una soglia fissata su train/validation)."""
    y_test = np.asarray(y_test).astype(int)
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError(f"{model}: punteggi non finiti")
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    auc, lo, hi = auroc_ci(y_test, score)
    row = {
        "model": model, "family": family, "seed": seed, "split": split,
        "n_train": n_train, "n_params": n_params,
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
        "average_precision": average_precision_score(y_test, score),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "fp": int(fp), "fn": int(fn), "threshold": threshold,
        "train_seconds": train_seconds, "infer_seconds": infer_seconds, "notes": notes,
    }
    for p in PFA_LEVELS:
        row[f"pd_at_pfa_{p:g}"] = pd_at_pfa(y_test, score, p)
    return {k: row[k] for k in RESULT_COLUMNS}


def append_results(path: str | Path, rows: Sequence[dict]) -> Path:
    """Aggiunge righe al CSV dei risultati (lo crea con l'intestazione se manca)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in RESULT_COLUMNS})
    return path
