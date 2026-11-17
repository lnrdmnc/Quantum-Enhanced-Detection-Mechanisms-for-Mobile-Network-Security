"""Classificatori variazionali (famiglia B) con lo stesso protocollo delle baseline.

Circuiti portati dai notebook originali:

* ``qaoa_rf_aware``  — notebook QAOA v2: RZ per qubit, IsingZZ per arco scalato dalla
  correlazione delle feature nel train, RX per qubit, ripetuti per layer;
* ``vqe_sel``        — notebook Comparing: StronglyEntanglingLayers ("VQE-inspired");
* ``qaoa_global``    — notebook Comparing: RZ(γ·x), IsingZZ(γ) a catena, RX(2β), 2 parametri per layer.

Tutti i circuiti accettano un *batch* di campioni (broadcasting di PennyLane): una
sola chiamata per minibatch invece di una per campione, ordini di grandezza più veloce.

Lettura del qubit 0 (z = ⟨Z₀⟩ ∈ [-1, 1]):

* ``readout="prob"``   p = (1 − z)/2, come nel notebook Comparing (nessun parametro);
* ``readout="affine"`` p = σ(a·z + b) con a, b addestrabili, come nel QAOA v2.

Ottimizzatori: ``"adam"`` oppure ``"qng_diag"`` (VQS-inspired: gradiente diviso per la
diagonale del tensore metrico, cioè quantum natural gradient diagonale, che è la forma
che assume la VQS in tempo immaginario). Il modello scelto è quello con la migliore
AUROC di validazione; il test non viene mai usato durante l'addestramento.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Circuiti
# ---------------------------------------------------------------------------
@dataclass
class Circuit:
    name: str
    fn: Callable          # fn(params_circuito, X) -> ⟨Z0⟩, shape (B,) o scalare
    n_params: int
    n_qubits: int
    info: dict = field(default_factory=dict)


def edges_from_correlation(X_train: np.ndarray, n_extra: Optional[int] = None):
    """Catena (i, i+1) + gli ``n_extra`` archi con |correlazione| più alta nel train
    (default: tanti quanti i qubit), come nel notebook QAOA v2."""
    n = X_train.shape[1]
    cor = np.nan_to_num(np.corrcoef(X_train.T))
    chain = {(i, i + 1) for i in range(n - 1)}
    others = sorted(((i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in chain),
                    key=lambda p: abs(cor[p]), reverse=True)[: (n if n_extra is None else n_extra)]
    edges = sorted(chain | set(others))
    couplings = np.array([np.clip(cor[i, j], -1, 1) for i, j in edges])
    return edges, couplings


def _device(n_qubits: int, device: str):
    return qml.device(device, wires=n_qubits)


def qaoa_rf_aware(n_qubits: int, layers: int, edges, couplings, device: str = "default.qubit") -> Circuit:
    dev = _device(n_qubits, device)
    edges = list(edges)
    couplings = np.asarray(couplings, dtype=float)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def fn(w, X):
        qml.AngleEmbedding(X, wires=range(n_qubits), rotation="Y")
        k = 0
        for _ in range(layers):
            for i in range(n_qubits):
                qml.RZ(w[k], wires=i); k += 1
            for e, (i, j) in enumerate(edges):
                qml.IsingZZ(w[k] * couplings[e], wires=[i, j]); k += 1
            for i in range(n_qubits):
                qml.RX(w[k], wires=i); k += 1
        return qml.expval(qml.PauliZ(0))

    return Circuit("qaoa_rf_aware", fn, layers * (2 * n_qubits + len(edges)), n_qubits,
                   {"layers": layers, "edges": edges, "couplings": couplings.tolist()})


def vqe_sel(n_qubits: int, layers: int, device: str = "default.qubit") -> Circuit:
    dev = _device(n_qubits, device)
    shape = (layers, n_qubits, 3)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def fn(w, X):
        qml.AngleEmbedding(X, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(pnp.reshape(w, shape), wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return Circuit("vqe_sel", fn, int(np.prod(shape)), n_qubits, {"layers": layers})


def qaoa_global(n_qubits: int, layers: int, device: str = "default.qubit") -> Circuit:
    dev = _device(n_qubits, device)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def fn(w, X):
        qml.AngleEmbedding(X, wires=range(n_qubits), rotation="Y")
        Xt = qml.math.transpose(X) if qml.math.ndim(X) == 2 else X  # (n_qubits, B) o (n_qubits,)
        for layer in range(layers):
            g, b = w[layer], w[layers + layer]
            for i in range(n_qubits):
                qml.RZ(g * Xt[i], wires=i)
            for i in range(n_qubits - 1):
                qml.IsingZZ(g, wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RX(2 * b, wires=i)
        return qml.expval(qml.PauliZ(0))

    return Circuit("qaoa_global", fn, 2 * layers, n_qubits, {"layers": layers})


# ---------------------------------------------------------------------------
# Lettura e loss
# ---------------------------------------------------------------------------
_EPS = 1e-7


def _split(w, circuit: Circuit):
    return w[: circuit.n_params], w[circuit.n_params:]


def _logit_or_prob(w, circuit: Circuit, X, readout: str):
    wc, wr = _split(w, circuit)
    z = circuit.fn(wc, X)
    if readout == "affine":
        return wr[0] * z + wr[1]
    return _EPS + (1 - 2 * _EPS) * (1 - z) / 2


def loss(w, circuit: Circuit, X, y, readout: str):
    """Cross-entropy binaria media sul batch."""
    out = _logit_or_prob(w, circuit, X, readout)
    y = np.asarray(y, dtype=float)
    if readout == "affine":
        t = 2 * y - 1
        return pnp.mean(pnp.logaddexp(0.0, -t * out))
    return -pnp.mean(y * pnp.log(out) + (1 - y) * pnp.log(1 - out))


def predict_score(w, circuit: Circuit, X, readout: str, batch: int = 2000) -> np.ndarray:
    """Probabilità di classe malevola (alto = malevolo), a blocchi."""
    w = pnp.array(np.asarray(w), requires_grad=False)
    out = []
    for i in range(0, len(X), batch):
        o = np.asarray(_logit_or_prob(w, circuit, np.asarray(X[i:i + batch]), readout))
        out.append(1 / (1 + np.exp(-o)) if readout == "affine" else o)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Addestramento
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    params: np.ndarray
    history: list
    steps_run: int
    train_seconds: float
    best_val_auc: float
    best_step: int


def _balanced_batch(y, size, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    h = size // 2
    b = np.r_[rng.choice(pos, h, replace=len(pos) < h), rng.choice(neg, size - h, replace=len(neg) < size - h)]
    rng.shuffle(b)
    return b


def train(
    circuit: Circuit,
    X_train, y_train, X_val, y_val,
    *,
    readout: str = "affine",
    optimizer: str = "adam",
    steps: int = 1200,
    batch: int = 32,
    lr: float = 0.015,
    eval_every: int = 25,
    patience: int = 12,
    init_scale: float = 0.12,
    metric_every: int = 20,
    metric_samples: int = 4,
    qng_damping: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """Minibatch bilanciati, early stopping sull'AUROC di validazione (ogni ``eval_every`` passi)."""
    if readout not in ("affine", "prob") or optimizer not in ("adam", "qng_diag"):
        raise ValueError("readout: 'affine'|'prob'; optimizer: 'adam'|'qng_diag'")
    rng = np.random.default_rng(seed)
    init = rng.normal(0, init_scale, circuit.n_params)
    if readout == "affine":
        init = np.r_[init, 2.0, 0.0]
    w = pnp.array(init, requires_grad=True)
    y_train = np.asarray(y_train).astype(int)
    opt = qml.AdamOptimizer(stepsize=lr) if optimizer == "adam" else None
    metric_fn = qml.metric_tensor(circuit.fn, approx="diag") if optimizer == "qng_diag" else None
    diag = None

    best_auc, best_w, best_step, wait, history = -np.inf, np.array(w), 0, 0, []
    t0 = time.perf_counter()
    step = 0
    for step in range(1, steps + 1):
        idx = _balanced_batch(y_train, batch, rng)
        xb, yb = np.asarray(X_train[idx]), y_train[idx]
        f = lambda v: loss(v, circuit, xb, yb, readout)  # noqa: E731
        if opt is not None:
            w = opt.step(f, w)
        else:
            g = np.asarray(qml.grad(f)(w))
            if diag is None or (step - 1) % metric_every == 0:
                wc = pnp.array(np.asarray(w[: circuit.n_params]), requires_grad=True)
                diags = []
                for x in xb[:metric_samples]:
                    m = np.asarray(metric_fn(wc, x))
                    diags.append(np.diag(m.reshape(circuit.n_params, circuit.n_params))
                                 if m.size == circuit.n_params ** 2 else m.reshape(-1))
                diag = np.maximum(np.mean(diags, axis=0), 1e-3)
            pre = np.ones(len(g))
            pre[: circuit.n_params] = diag + qng_damping
            w = pnp.array(np.asarray(w) - lr * g / pre, requires_grad=True)
        if step % eval_every == 0 or step == steps:
            auc = roc_auc_score(y_val, predict_score(w, circuit, X_val, readout))
            history.append({"step": step, "val_auc": float(auc)})
            if verbose:
                print(f"  step {step:5d}  val AUROC {auc:.4f}", flush=True)
            if auc > best_auc + 1e-4:
                best_auc, best_w, best_step, wait = auc, np.array(w), step, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    return TrainResult(best_w, history, step, time.perf_counter() - t0, float(best_auc), best_step)


def n_trainable(circuit: Circuit, readout: str) -> int:
    return circuit.n_params + (2 if readout == "affine" else 0)
