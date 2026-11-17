"""Baseline classiche con lo stesso protocollo dei modelli quantistici.

Ogni modello restituisce un punteggio "alto = malevolo" sul test; ``run_baselines``
addestra su ``train`` (5.000 righe, come i circuiti) e/o ``train_large``
(20.000 righe) per più seed e scrive righe nel formato di ``qdetect.metrics``.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .data import Prepared
from .metrics import evaluate

# nome → (costruttore(seed), stocastico?)
MODELS: dict[str, tuple[Callable[[int], object], bool]] = {
    "Logistic Regression": (lambda s: LogisticRegression(max_iter=5000), False),
    # MLP "piccolo": 8→10→1 ≈ 101 parametri, ordine di grandezza dei circuiti (~100)
    "MLP small (10)": (lambda s: MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000,
                                                early_stopping=True, random_state=s), True),
    # stessa architettura di prova.py
    "MLP (100, 50)": (lambda s: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                                               early_stopping=True, random_state=s), True),
    "SVM RBF": (lambda s: SVC(kernel="rbf", C=1.0, gamma="scale"), False),
    "Random Forest": (lambda s: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                       n_jobs=-1, random_state=s), True),
    "Gradient Boosting": (lambda s: HistGradientBoostingClassifier(random_state=s), True),
}


def n_params(model) -> Optional[int]:
    """Numero di parametri addestrabili per i modelli parametrici (None per alberi e SVM)."""
    if isinstance(model, MLPClassifier):
        return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    return None


def score(model, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Punteggio "alto = malevolo" e soglia di decisione corrispondente."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], 0.5
    return model.decision_function(X), 0.0


def run_baselines(
    P: Prepared,
    models: Optional[Iterable[str]] = None,
    train_sets: Iterable[str] = ("train", "train_large"),
    seeds: Iterable[int] = (42, 123, 2024, 2025, 2026),
    eval_sets: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Addestra e valuta. ``eval_sets`` default: ``{'test': (P.X_test, P.y_test)}`` più
    gli insiemi in ``P.extra``. I modelli deterministici girano con un solo seed."""
    names = list(models) if models is not None else list(MODELS)
    if eval_sets is None:
        eval_sets = {"test": (P.X_test, P.y_test), **P.extra}
    data = {"train": (P.X_train, P.y_train), "train_large": (P.X_train_large, P.y_train_large)}
    rows = []
    for name in names:
        build, stochastic = MODELS[name]
        for ts in train_sets:
            Xtr, ytr = data[ts]
            for seed in (list(seeds) if stochastic else [list(seeds)[0]]):
                m = build(seed)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                t_fit = time.perf_counter() - t0
                for es, (Xe, ye) in eval_sets.items():
                    t0 = time.perf_counter()
                    s, thr = score(m, Xe)
                    t_inf = time.perf_counter() - t0
                    rows.append(evaluate(name, ye, s, family="classical", seed=seed, split=es,
                                         n_train=len(ytr), n_params=n_params(m), threshold=thr,
                                         train_seconds=t_fit, infer_seconds=t_inf,
                                         notes=f"train_set={ts}"))
                if verbose:
                    last = rows[-len(eval_sets):]
                    print(f"{name:20s} {ts:12s} seed={seed:<5d} "
                          + "  ".join(f"{r['split']}={r['auroc']:.4f}" for r in last)
                          + f"  fit={t_fit:.1f}s", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Media e deviazione standard sui seed, per modello, insieme di train e di test."""
    d = df.assign(train_set=df["notes"].str.replace("train_set=", "", regex=False))
    cols = ["auroc", "f1", "pd_at_pfa_0.01", "pd_at_pfa_0.05", "train_seconds"]
    out = d.groupby(["split", "train_set", "model"])[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    n = d.groupby(["split", "train_set", "model"])["seed"].nunique().rename("n_seeds")
    return out.join(n).reset_index().sort_values(["split", "train_set", "auroc_mean"],
                                                  ascending=[True, True, False])
