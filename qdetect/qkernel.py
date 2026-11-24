"""Kernel quantistico di fedeltà simulato esattamente (statevector, solo numpy).

Feature map della specifica del relatore: RY(x_j) su ogni qubit, anello di CNOT,
re-uploading RY(0.7·x_j). Il controllo *separabile* usa le stesse rotazioni senza CNOT.

    K_Q(x, x') = |⟨ψ(x)|ψ(x')⟩|²

Gli stati hanno ampiezze reali (solo RY e CNOT), quindi il prodotto scalare è reale.

* ``geometric_difference``: g_CQ = sqrt(λ_max(K_C^{-1/2} K_Q K_C^{-1/2})) con
  Tr(K_C) = Tr(K_Q) = N e una regolarizzazione (ridge) dichiarata (Huang et al., 2021).
  È un *prescreening* geometrico, non una prova di vantaggio.
* ``hoeffding_shots``: shot per stimare ogni overlap con errore ε (bound di
  Hoeffding con union bound sulle coppie): una stima di costo, non un limite inferiore.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def states(X: np.ndarray, entangled: bool = True, reupload: float = 0.7) -> np.ndarray:
    """Stati |ψ(x)⟩ per un batch di vettori di angoli, shape (N, 2^q)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    N, q = X.shape
    dim = 2 ** q
    idx = np.arange(dim)
    v = np.zeros((N, dim))
    v[:, 0] = 1.0

    def ry(v, j, t):
        lo = idx[(idx & (1 << j)) == 0]
        hi = lo | (1 << j)
        c, s = np.cos(t / 2)[:, None], np.sin(t / 2)[:, None]
        out = v.copy()
        out[:, lo] = c * v[:, lo] - s * v[:, hi]
        out[:, hi] = s * v[:, lo] + c * v[:, hi]
        return out

    for j in range(q):
        v = ry(v, j, X[:, j])
    if entangled:
        for j in range(q):  # CNOT controllo j, bersaglio j+1 (anello)
            k = (j + 1) % q
            perm = idx ^ (((idx >> j) & 1) << k)
            v = v[:, perm]
    for j in range(q):
        v = ry(v, j, reupload * X[:, j])
    return v


def fidelity_kernel(A: np.ndarray, B: np.ndarray | None = None, entangled: bool = True) -> np.ndarray:
    SA = states(A, entangled)
    SB = SA if B is None else states(B, entangled)
    return (SA @ SB.T) ** 2


def _normalize(K: np.ndarray) -> np.ndarray:
    return K * len(K) / np.trace(K)


def geometric_difference(K_q: np.ndarray, K_c: np.ndarray, ridge: float = 1e-4) -> float:
    """g_CQ con tracce normalizzate a N e ridge simmetrico su entrambe le matrici."""
    Kq, Kc = _normalize(K_q), _normalize(K_c)
    n = len(Kq)
    lam = eigh(Kq + ridge * np.eye(n), Kc + ridge * np.eye(n), eigvals_only=True)[-1]
    return float(np.sqrt(max(0.0, lam)))


def effective_rank(K: np.ndarray) -> float:
    """(Σλ)² / Σλ²: numero "efficace" di direzioni del kernel."""
    ev = np.clip(np.linalg.eigvalsh(K), 0, None)
    return float(ev.sum() ** 2 / (ev @ ev))


def offdiag_stats(K: np.ndarray) -> dict:
    """Media e deviazione standard fuori diagonale: valori tutti vicini fra loro
    indicano concentrazione esponenziale (Thanasilp et al., 2024)."""
    off = K[np.triu_indices(len(K), 1)]
    return {"offdiag_mean": float(off.mean()), "offdiag_std": float(off.std())}


def hoeffding_shots(n_pairs: int, eps: float, family_failure: float = 0.05) -> int:
    """Shot per coppia perché tutti gli overlap siano entro ε con probabilità 1−δ."""
    return int(np.ceil(np.log(2 * n_pairs / family_failure) / (2 * eps ** 2)))
