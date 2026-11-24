# QCA9880: jamming debole (notebook 62, eseguito sul server il 24/11/2026)

Corpus: `monaco/spectral_scans_QCA9880`, larghezza ht20, file processati; manifest `monaco/inventory_full.csv`.
- 2.250 file di jamming in camera, su 25 canali (2,4 e 5 GHz), a 20/40/60 cm e 0/5/10 dBm, 10 ripetizioni ciascuno;
- 30 file benigni: 10 in camera, 10 in laboratorio, 10 in ufficio.

Ogni file è una scansione di tutti i 25 canali, con circa 460–700 righe per canale.
I risultati completi sono sul server in `.../results/qca9880/`.

## 1. Benigni dedicati: c'è una scorciatoia di sessione
Con i 10 benigni in camera come negativi, anche i canali **lontani dal jammer** separano le classi (AUROC mediana 0,994). I file benigni vengono da un'altra sessione, quindi quel confronto non misura il jamming. Inoltre, con soglia al 99° percentile dei benigni in camera, laboratorio e ufficio danno il 59–79% di falsi allarmi.

## 2. Disegno pulito: negativi della stessa campagna
Per ogni canale f, i negativi sono le finestre a f dei file con il jammer nell'**altra banda**; i positivi sono quelli con il jammer su f.
Split per file 50/25/25, Mahalanobis con shrinkage, soglia conforme: il falso allarme osservato è ≈ 1% (0,9–1,1%), quindi la calibrazione regge.

AUROC media sui canali a 5 GHz, distanza 60 cm (a 20 e 40 cm, e a 2,4 GHz, è quasi sempre ≥ 0,98):

| W righe | 0 dBm | 5 dBm | 10 dBm |
|---|---|---|---|
| 8 | 0,768 | 0,876 | 0,991 |
| 32 | 0,815 | 0,979 | 1,000 |
| 128 | 0,828 | 0,991 | 1,000 |

Rilevamento a α = 1%, 60 cm, 5 GHz: a 0 dBm 32% (W = 8) → 44% (W = 128); a 5 dBm 58% → 75%.
Canali più difficili a 60 cm, 0 dBm, W = 8: 5260 (AUROC 0,44), 5825 (0,45), 5845 (0,59), 5280 (0,67) MHz.

**È il primo regime davvero difficile trovato:** jamming debole (0 dBm a 60 cm) su alcuni canali a 5 GHz, con l'effetto del budget di report ben visibile.

Limite: per ogni combinazione (canale, distanza, potenza) i file di test sono pochi (circa 2–3, una finestra per file). I valori per singolo canale sono molto incerti; quelli mediati sui canali sono più stabili. Usare più finestre per file e più ripetizioni è il passo successivo.
