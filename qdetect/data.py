"""Dati comuni per tutti i modelli: campionamento, split congelato, preprocessing.

Flusso tipico
-------------
1. ``sample_per_class``  legge il CSV (anche da GB) a blocchi ed estrae un campione
   uniforme senza reinserimento per ciascuna classe.
2. ``make_split``        divide il campione in train / validation / test senza
   vettori di feature condivisi tra gli insiemi, poi estrae i sotto-insiemi
   stratificati di dimensione fissa usati da tutti i modelli.
3. ``save_split``        congela tutto in un ``.npz``: ogni notebook carica lo stesso
   file con ``load_split`` e vede esattamente le stesse righe.
4. ``prepare``           stima imputazione, standardizzazione (ed eventuale
   selezione di feature / mappatura angolare) solo sul training.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES: list[str] = [
    "freq1", "noise", "max_magnitude", "total_gain_db",
    "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db",
]
TARGET = "label_encoded"
SPLIT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# 1. Campionamento dal CSV
# ---------------------------------------------------------------------------
def sample_per_class(
    path: str | Path,
    cap_per_class: int,
    seed: int = 2026,
    features: Sequence[str] = FEATURES,
    target: str = TARGET,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict]:
    """Campione uniforme senza reinserimento di ``cap_per_class`` righe per classe.

    A ogni riga del file si associa una chiave casuale U(0,1) e si tengono, per
    classe, le ``cap`` righe con chiave più piccola: equivale a un reservoir
    sampling ma è vettorizzato. Le chiavi sono estratte per *tutte* le righe nello
    stesso ordine, quindi il risultato dipende solo da ``seed`` e dal contenuto del
    file, non da ``chunksize``.

    Restituisce il DataFrame (feature, ``target``, ``row_id`` = posizione della
    riga nel file, header escluso) ordinato per ``row_id``, e le statistiche di
    lettura.
    """
    features = list(features)
    rng = np.random.default_rng(seed)
    kept: dict[int, Optional[pd.DataFrame]] = {0: None, 1: None}
    seen = {0: 0, 1: 0}
    n_rows = n_dropped = 0

    reader = pd.read_csv(path, usecols=features + [target], chunksize=chunksize,
                         low_memory=False)
    for chunk in reader:
        keys = rng.random(len(chunk))  # prima di qualsiasi filtro: indipendente dal chunksize
        chunk = chunk.copy()
        chunk["row_id"] = np.arange(n_rows, n_rows + len(chunk), dtype=np.int64)
        chunk["_key"] = keys
        n_rows += len(chunk)

        y = pd.Series(pd.to_numeric(chunk[target], errors="coerce"), index=chunk.index)
        ok = y.isin([0, 1])
        n_dropped += int((~ok).sum())
        chunk = chunk.loc[ok]
        y = y.loc[ok].astype(int)

        for c in (0, 1):
            sub = chunk.loc[y == c]
            seen[c] += len(sub)
            merged = sub if kept[c] is None else pd.concat([kept[c], sub], ignore_index=True)
            if len(merged) > cap_per_class:
                merged = merged.nsmallest(cap_per_class, "_key")
            kept[c] = merged

    if min(seen.values()) == 0:
        raise ValueError(f"Classe assente o etichette non 0/1: {seen}")
    for c in (0, 1):
        if seen[c] < cap_per_class:
            warnings.warn(f"Classe {c}: solo {seen[c]} righe (< cap {cap_per_class}), prese tutte.")

    df = pd.concat([kept[0], kept[1]], ignore_index=True).drop(columns="_key")
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[target] = df[target].astype(int)
    df = df.sort_values("row_id").reset_index(drop=True)
    stats = {"rows_in_file": n_rows, "rows_per_class_in_file": seen,
             "rows_dropped_bad_label": n_dropped, "sample_seed": seed,
             "cap_per_class": cap_per_class}
    return df, stats


# ---------------------------------------------------------------------------
# 2. Diagnostica sul campione
# ---------------------------------------------------------------------------
def feature_groups(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Identificativo (hash uint64) del vettore di feature: righe identiche → stesso id."""
    frame = pd.DataFrame(np.asarray(X))
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def duplicate_report(X: np.ndarray, y: np.ndarray) -> dict:
    """Duplicati e conflitti di etichetta.

    ``bayes_error_floor``: frazione minima di errori di *qualunque* classificatore
    che usi solo queste feature, dovuta ai vettori identici con etichette diverse
    (per ogni gruppo si sbaglia almeno la classe minoritaria). È un limite
    inferiore: con feature continue vicine l'errore irriducibile può essere più alto.
    """
    g = feature_groups(X)
    tab = pd.crosstab(g, np.asarray(y))
    for c in (0, 1):
        if c not in tab.columns:
            tab[c] = 0
    sizes = tab[0] + tab[1]
    conflict = (tab[0] > 0) & (tab[1] > 0)
    n = len(y)
    return {
        "rows": int(n),
        "unique_vectors": int(len(tab)),
        "duplicate_row_fraction": float(1 - len(tab) / n),
        "conflicting_vectors": int(conflict.sum()),
        "rows_in_conflicting_vectors": int(sizes[conflict].sum()),
        "bayes_error_floor": float(np.minimum(tab[0], tab[1]).sum() / n),
    }


# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """Campione congelato + indici (posizioni in ``X``/``y``) di ogni insieme."""

    X: np.ndarray                 # (N, n_features) feature grezze, float
    y: np.ndarray                 # (N,) etichette 0/1
    row_id: np.ndarray            # (N,) posizione della riga nel CSV originale
    features: list[str]
    idx_train: np.ndarray         # training comune (modelli quantistici, RF Small, ...)
    idx_train_large: np.ndarray   # superset di idx_train (RF Large, baseline con più dati)
    idx_val: np.ndarray
    idx_test: np.ndarray
    meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "train_large", "val", "test"):
            idx = getattr(self, f"idx_{name}")
            rows.append({"set": name, "n": len(idx), "frac_malicious": float(self.y[idx].mean())})
        return pd.DataFrame(rows)


def _stratified_take(idx: np.ndarray, y: np.ndarray, n: int, seed: int, name: str) -> np.ndarray:
    idx = np.asarray(idx)
    if n >= len(idx):
        if n > len(idx):
            warnings.warn(f"{name}: richiesti {n}, disponibili {len(idx)}: presi tutti.")
        return np.sort(idx)
    take, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)
    return np.sort(take)


def _group_split(groups: np.ndarray, y: np.ndarray, test_frac: float, val_frac: float, seed: int):
    all_idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trval, test = next(gss.split(all_idx, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed + 1)
    a, b = next(gss2.split(trval, y[trval], groups[trval]))
    return trval[a], trval[b], test


def _block_split(groups: np.ndarray, row_id: np.ndarray, test_frac: float, val_frac: float,
                 seed: int, n_blocks: int):
    """Blocchi contigui di ``row_id`` assegnati a caso agli insiemi, poi *purge*.

    Utile se il CSV è (anche solo in parte) in ordine di acquisizione: righe vicine
    nel file non finiscono in insiemi diversi, salvo ai bordi dei blocchi. Dopo
    l'assegnazione si eliminano da val/test i vettori già presenti nel train (e dal
    test quelli presenti in val), così resta garantita l'assenza di duplicati esatti.
    """
    order = np.argsort(row_id, kind="stable")
    blocks = np.array_split(order, n_blocks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    n_test = max(1, round(test_frac * n_blocks))
    n_val = max(1, round(val_frac * n_blocks))
    test = np.concatenate([blocks[i] for i in perm[:n_test]])
    val = np.concatenate([blocks[i] for i in perm[n_test:n_test + n_val]])
    train = np.concatenate([blocks[i] for i in perm[n_test + n_val:]])
    g_train = set(groups[train].tolist())
    val = val[~np.isin(groups[val], list(g_train))]
    g_trval = g_train | set(groups[val].tolist())
    test = test[~np.isin(groups[test], list(g_trval))]
    return train, val, test


def make_split(
    df: pd.DataFrame,
    n_train: int = 5_000,
    n_train_large: int = 20_000,
    n_val: int = 1_000,
    n_test: int = 5_000,
    strategy: str = "group",
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    n_blocks: int = 200,
    seed: int = 42,
    features: Sequence[str] = FEATURES,
    target: str = TARGET,
) -> Split:
    """Split senza vettori di feature condivisi + sotto-insiemi stratificati.

    ``strategy``:
      * ``"group"``: GroupShuffleSplit sui vettori identici (come nei notebook attuali);
      * ``"block"``: blocchi contigui del file + purge dei duplicati (controllo di
        robustezza contro la correlazione tra righe vicine).

    ``idx_train`` è contenuto in ``idx_train_large``: il confronto "a parità di
    campioni" (es. RF Small vs modelli quantistici) usa esattamente le stesse righe.
    """
    features = list(features)
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    row_id = df["row_id"].to_numpy() if "row_id" in df else np.arange(len(df))
    groups = feature_groups(X)

    if strategy == "group":
        tr_pool, va_pool, te_pool = _group_split(groups, y, test_frac, val_frac, seed)
    elif strategy == "block":
        tr_pool, va_pool, te_pool = _block_split(groups, row_id, test_frac, val_frac, seed, n_blocks)
    else:
        raise ValueError("strategy deve essere 'group' o 'block'")

    if n_train > n_train_large:
        raise ValueError("n_train deve essere <= n_train_large")
    idx_train_large = _stratified_take(tr_pool, y, n_train_large, seed + 10, "train_large")
    idx_train = _stratified_take(idx_train_large, y, n_train, seed + 11, "train")
    idx_val = _stratified_take(va_pool, y, n_val, seed + 12, "val")
    idx_test = _stratified_take(te_pool, y, n_test, seed + 13, "test")

    split = Split(
        X=X, y=y, row_id=row_id, features=features,
        idx_train=idx_train, idx_train_large=idx_train_large,
        idx_val=idx_val, idx_test=idx_test,
        meta={"strategy": strategy, "seed": seed, "test_frac": test_frac,
              "val_frac": val_frac, "n_blocks": n_blocks if strategy == "block" else None,
              "pool_sizes": {"train": len(tr_pool), "val": len(va_pool), "test": len(te_pool)}},
    )
    check_split(split)
    return split


def check_split(split: Split) -> None:
    """Asserzioni di integrità: nessuna riga e nessun vettore condiviso tra insiemi."""
    g = feature_groups(split.X)
    tr, trl, va, te = (split.idx_train, split.idx_train_large, split.idx_val, split.idx_test)
    assert set(tr) <= set(trl), "idx_train deve stare dentro idx_train_large"
    for a_name, a, b_name, b in [("train_large", trl, "val", va), ("train_large", trl, "test", te),
                                 ("val", va, "test", te)]:
        assert not (set(a) & set(b)), f"righe condivise tra {a_name} e {b_name}"
        assert not (set(g[a]) & set(g[b])), f"vettori identici tra {a_name} e {b_name}"


# ---------------------------------------------------------------------------
# 4. Salvataggio / caricamento
# ---------------------------------------------------------------------------
def save_split(split: Split, path: str | Path, extra_meta: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**split.meta, **(extra_meta or {}), "features": split.features,
            "format_version": SPLIT_FORMAT_VERSION}
    np.savez_compressed(
        path, X=split.X, y=split.y, row_id=split.row_id,
        idx_train=split.idx_train, idx_train_large=split.idx_train_large,
        idx_val=split.idx_val, idx_test=split.idx_test,
        meta=np.array(json.dumps(meta, default=str)),
    )
    return path


def load_split(path: str | Path) -> Split:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format_version") != SPLIT_FORMAT_VERSION:
            raise ValueError(f"Versione split non supportata: {meta.get('format_version')}")
        split = Split(X=z["X"], y=z["y"], row_id=z["row_id"], features=list(meta["features"]),
                      idx_train=z["idx_train"], idx_train_large=z["idx_train_large"],
                      idx_val=z["idx_val"], idx_test=z["idx_test"], meta=meta)
    check_split(split)
    return split


# ---------------------------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    """Matrici pronte per i modelli; il preprocessore è stimato solo su ``fit_on``."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_train_large: np.ndarray
    y_train_large: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    features: list[str]
    preprocessor: Pipeline
    config: dict


def angle_map(Z: np.ndarray, scale: float) -> np.ndarray:
    """Feature standardizzate → angoli in (-π, π) con π·tanh(z/scale)."""
    return np.pi * np.tanh(np.asarray(Z) / scale)


def prepare(
    split: Split,
    features: Optional[Sequence[str]] = None,
    k_best: Optional[int] = None,
    angle_scale: Optional[float] = None,
    fit_on: str = "train",
) -> Prepared:
    """Imputer mediana → StandardScaler → (SelectKBest) → (π·tanh(z/angle_scale)).

    * ``features``: sottoinsieme di colonne (default: tutte quelle dello split);
    * ``k_best``: selezione supervisionata con ANOVA F, stimata solo su ``fit_on``;
    * ``angle_scale``: ``None`` = feature standardizzate; un numero = angoli per
      l'angle embedding. Usare lo stesso valore per tutti i modelli confrontati;
    * ``fit_on``: ``"train"`` (default) o ``"train_large"``.
    """
    features = list(features) if features is not None else list(split.features)
    cols = [split.features.index(f) for f in features]
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if k_best is not None:
        steps.append(("selector", SelectKBest(f_classif, k=k_best)))
    pre = Pipeline(steps)

    fit_idx = {"train": split.idx_train, "train_large": split.idx_train_large}[fit_on]
    pre.fit(split.X[np.ix_(fit_idx, cols)], split.y[fit_idx])
    selected = (np.array(features)[pre.named_steps["selector"].get_support()].tolist()
                if k_best is not None else features)

    def tf(idx):
        Z = pre.transform(split.X[np.ix_(idx, cols)])
        return angle_map(Z, angle_scale) if angle_scale is not None else Z

    return Prepared(
        X_train=tf(split.idx_train), y_train=split.y[split.idx_train],
        X_train_large=tf(split.idx_train_large), y_train_large=split.y[split.idx_train_large],
        X_val=tf(split.idx_val), y_val=split.y[split.idx_val],
        X_test=tf(split.idx_test), y_test=split.y[split.idx_test],
        features=selected, preprocessor=pre,
        config={"features_in": features, "features_out": selected, "k_best": k_best,
                "angle_scale": angle_scale, "fit_on": fit_on,
                "split_strategy": split.meta.get("strategy")},
    )
