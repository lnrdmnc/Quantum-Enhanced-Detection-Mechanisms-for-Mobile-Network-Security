"""Manifest del dataset per file (WiFiSpectralJam / RF Jamming Dataset) ricavato dai percorsi.

La copia sul server non contiene il ``manifest.csv`` della release 2026: etichetta,
modalità di scansione, sottotipo benigno, forma d'onda, potenza, canale e luogo sono
però codificati in cartelle e nomi di file::

    active_scan/malicious/{gaussian_noise|singletone}/{neg40dbm|neg10dbm|0dbm|10dbm}/5805mhz_jamming_<p>_<wf>_<i>.csv
    active_scan/benign/floor/5ghz_activescan_floor_<i>.csv
    active_scan/benign/background/location{1,2}/5ghz_activescan_background_loc<k>_<i>.csv
    passive_scan/malicious/gaussian_noise/{3,6,9,12}dbm/<f>mhz_jamming_<p>_gaussiannoise_<i>.csv
    passive_scan/benign/floor/{2.4ghz|5ghz}_passivescan_floor_<i>.csv
    passive_scan/benign/background/location{1,2,3}/<band>_passivescan_background_loc<k>_<i>.csv

I file in ``.ipynb_checkpoints`` sono copie di Jupyter e vengono esclusi.
Le colonne seguono lo schema richiesto dal notebook del relatore
(``file_path, scan_mode, label, benign_subtype, waveform, power_dbm, channel_mhz``)
più ``band`` e ``location``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = ["file_path", "scan_mode", "label", "benign_subtype", "waveform",
                    "power_dbm", "channel_mhz", "band", "location"]

_POWER = re.compile(r"^(neg)?(\d+)dbm$")
_MAL = re.compile(r"^(\d+)mhz_jamming_")
_BAND = re.compile(r"^(2\.4|5)ghz_")


def _power(token: str) -> float:
    m = _POWER.match(token)
    if not m:
        raise ValueError(f"potenza non riconosciuta: {token}")
    return -float(m.group(2)) if m.group(1) else float(m.group(2))


def parse_path(rel: str) -> dict:
    """Metadati di un file a partire dal percorso relativo alla radice del dataset."""
    parts = Path(rel).parts
    name = parts[-1]
    if len(parts) < 4 or not parts[0].endswith("_scan"):
        raise ValueError(f"percorso inatteso: {rel}")
    row = {"file_path": rel, "scan_mode": parts[0].replace("_scan", ""), "label": parts[1],
           "benign_subtype": "", "waveform": "", "power_dbm": np.nan, "channel_mhz": np.nan,
           "band": "", "location": ""}
    if row["label"] == "malicious":
        row["waveform"] = {"singletone": "single_tone", "gaussian_noise": "gaussian_noise"}[parts[2]]
        row["power_dbm"] = _power(parts[3])
        m = _MAL.match(name)
        if not m:
            raise ValueError(f"canale non riconosciuto: {name}")
        row["channel_mhz"] = float(m.group(1))
        row["band"] = "2.4ghz" if row["channel_mhz"] < 3000 else "5ghz"
        row["location"] = "chamber"
    elif row["label"] == "benign":
        row["benign_subtype"] = parts[2]
        if parts[2] == "floor":
            row["location"] = "chamber"
        elif parts[2] == "background":
            row["location"] = parts[3]
        else:
            raise ValueError(f"sottotipo benigno inatteso: {rel}")
        m = _BAND.match(name)
        if not m:
            raise ValueError(f"banda non riconosciuta: {name}")
        row["band"] = m.group(1) + "ghz"
    else:
        raise ValueError(f"etichetta inattesa: {rel}")
    return row


def build_manifest(root: str | Path) -> pd.DataFrame:
    """Percorre ``root`` e restituisce il manifest (un file = una riga), ordinato per percorso."""
    root = Path(root)
    rows = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]  # esclude .ipynb_checkpoints
        for f in files:
            if f.endswith(".csv") and not f.startswith("."):
                rows.append(parse_path(os.path.relpath(os.path.join(d, f), root)))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("file_path").reset_index(drop=True)
    if df.file_path.duplicated().any():
        raise ValueError("percorsi duplicati nel manifest")
    return df
