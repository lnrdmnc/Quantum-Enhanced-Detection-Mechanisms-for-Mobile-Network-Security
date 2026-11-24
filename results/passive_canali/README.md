# Passive scan per canale: risultati (notebook 61, eseguito sul server il 24/11/2026)

Dati: tutti i file passive malevoli (17.585), tutti i benigni di fondo in camera (20.432), 1.500 benigni di ambiente reale per banda e luogo. Split per file 50/25/25.
I risultati completi sono sul server in `monaco/Quantum-Enhanced-Detection-Mechanisms-for-Mobile-Network-Security/results/passive_canali/`.

## 1. Perché il 5745 MHz sembrava difficile
Ogni file passive è una scansione sequenziale dei canali: 5180 → 5825 MHz a 5 GHz, 2412 → 2462 MHz a 2,4 GHz, circa 4.700 righe. Il prefisso di 128 righe del protocollo contiene **solo il primo canale** (5180 o 2412 MHz).
L'AUROC 0,62 "sul canale 5745" del notebook 60 è quindi un effetto della finestra, non una difficoltà del jamming.

## 2. Sul canale del jammer il rilevamento è saturo
W = 128 righe al canale monitorato, rilevatore one-class addestrato sui benigni in camera, soglia conforme su circa 3.300 file benigni di calibrazione (1.760 a 2,4 GHz):
- AUROC 1,000 per tutti i canali e tutte le potenze, **anche a 3 dBm**;
- a α = 1%: TPR 1,000, FPR 0,7–1,4%;
- a α = 0,1%: TPR 1,000, FPR 0,00–0,15%, limite superiore al 95% 0,09–0,34%.

## 3. Budget di report (miglior modello one-class sulla diagonale)
| W | TPR a α = 0,1% (media per finestra) |
|---|---|
| 8 | 0,93–0,99 |
| 32 | 0,96–1,00 |
| 128 | 1,00 |

Le covarianze ("second") vanno peggio delle sole medie con poche righe. Il budget conta solo a rischio molto basso: è il candidato per la RQ4.

## 4. Scorciatoie (canali lontani dal jammer)
- **5 GHz:** AUROC one-class mediana 0,64–0,65 sui canali lontani dal jammer, supervisionata 0,91–0,94. Il Random Forest sfrutta piccole differenze di sessione.
- **2,4 GHz:** AUROC 1,000 su **tutti** i canali. Con jammer a 2412 MHz e canale monitorato a 2462 MHz (50 MHz di distanza), il solo `noise` separa le classi (AUROC 1,0: −96,1 dBm nei benigni, −101,9 dBm nei malevoli). È una differenza di acquisizione tra le sessioni benigne e malevole, oppure l'effetto del jammer sul ricevitore (AGC). I dati da soli non permettono di distinguere le due cause.

## 5. Fuori distribuzione: soglia calibrata in camera, benigni di ambiente reale
Falso allarme reale a α nominale 1%: **67% a 5 GHz e 99–100% a 2,4 GHz** per il one-class; 43–95% per il supervisionato.
I benigni in ambiente reale (traffico Wi-Fi vero) sembrano jamming a un rilevatore addestrato in camera. Il dataset non ha jamming registrato in ambiente reale, quindi questo regime, che è quello davvero difficile, non si può valutare con i dati attuali.
