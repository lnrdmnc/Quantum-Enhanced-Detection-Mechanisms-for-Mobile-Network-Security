# Protocollo del relatore sul dataset per file: risultati (esecuzione `full`)

Notebook: `notebooks/60_protocollo_file.ipynb`, eseguito sul server il 24/11/2026.
Dati: `monaco/rf_jamming_dataset` (96.090 file, manifest ricostruito dai percorsi).

Campione: 21.901 file, fino a 1.200 per strato; un prefisso di 128 report per file; split per hash del contenuto 50/25/25.

I risultati completi (`metrics.csv`, `sample_manifest.csv`, `attribution.csv`, matrici di Gram, notebook eseguito) sono sul server in:
`monaco/Quantum-Enhanced-Detection-Mechanisms-for-Mobile-Network-Security/results/protocollo_file/full/`.
Qui c'è solo `summary.json`.

## Sintesi

| Regime (training → test, benigni di test) | AUROC max one-class | AUROC max supervisionato | Note |
|---|---|---|---|
| active → active, fondo in camera | 1,000 (già con 8 report) | 1,000 | tutte le potenze, compreso −40 dBm, e entrambe le forme d'onda a 1,000 |
| active → active, ambiente reale | 0,976 | 1,000 | con `noise` e `total_gain_db`: 0,939 |
| passive → passive, fondo in camera | 0,876 | 0,976 | a FPR nominale 1%: TPR 0,73, FPR 0,7% (limite 95%: 2,2%) |
| passive → passive, ambiente reale | 0,622 | 0,917 | soglia calibrata sul fondo in camera: FPR reale 72% invece dell'1% |
| active → passive, fondo | 0,649 | 0,737 | trasferimento tra modalità difficile |
| passive → active, fondo | 0,995 | 0,882 | |

Passive, fondo in camera, 128 report, AUROC per canale (one-class / supervisionato):
2412 MHz 1,000 / 1,000 · 2457 MHz 0,999 / 0,999 · 5180 MHz 1,000 / 1,000 · **5745 MHz 0,624 / 0,916**.
Per potenza (3, 6, 9, 12 dBm) l'AUROC one-class è 0,89–0,90 ovunque: la difficoltà dipende dal canale, non dalla potenza.

Limiti emersi:
- **Active:** solo 404 file di fondo in camera, quindi circa 92 di calibrazione. Una soglia conforme a α = 1% richiede almeno 99 campioni: nessuna soglia è stimabile e lo stato è `INSUFFICIENT_RISK_EVIDENCE`.
- **Passive:** 329 file di calibrazione con il limite di 1.200 file per strato; per α = 0,1% ne servono almeno 999.

Geometric difference del kernel quantistico (6 qubit) rispetto all'RBF con γ = 1: 1,58–1,62 (active), 1,82–1,88 (passive). Valori simili a quelli ottenuti sul CSV.

Gate: `STOP_CONFIRMATORY__INDEPENDENT_SESSIONS_AND_RISK_REQUIRED`; nessun vantaggio quantistico.
