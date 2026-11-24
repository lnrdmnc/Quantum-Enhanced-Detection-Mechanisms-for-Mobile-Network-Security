# Regime di jamming debole: classici contro quantistici (notebook 63, eseguito sul server il 24/11/2026)

Corpus QCA9880, banda a 5 GHz, larghezza ht20.
- **Negativi:** finestre allo stesso canale dei file della stessa campagna con jammer a 2,4 GHz.
- **Positivi:** finestre al canale del jammer.
- Fino a 5 finestre per file e canale, split per file 50/25/25, feature standardizzate per canale sui negativi di training, un solo modello per la banda.
- **Test:** 16.660 finestre negative; a 60 cm e 0 dBm, 140 finestre positive da 28 file.
- **Soglie conformi sui negativi di calibrazione:** falso allarme osservato ≈ 1,0% e ≈ 0,1%, quindi la calibrazione regge.

I risultati completi sono sul server in `.../results/regime_debole/` (`metriche.csv`, `riepilogo_60cm_0dBm.csv`, `delong_60cm_0dBm.csv`, notebook eseguito).

## 60 cm, 0 dBm: AUROC (intervallo DeLong al 95%) e rilevamento a falso allarme fissato
| Modello | W = 8 | W = 32 | W = 128 | TPR a α = 1%, W = 8 → 128 | TPR a α = 0,1%, W = 8 → 128 |
|---|---|---|---|---|---|
| Random Forest (supervisionato) | 0,920 (0,89–0,95) | 0,962 | **0,979** (0,96–1,00) | 0,51 → 0,84 | 0,31 → 0,74 |
| MLP piccolo, 91 parametri | 0,871 | 0,924 | 0,953 | 0,47 → 0,70 | 0,32 → 0,55 |
| Regressione logistica | 0,867 | 0,909 | 0,938 | 0,47 → 0,67 | 0,29 → 0,53 |
| RFF-Mahalanobis (one-class) | 0,827 | 0,884 | 0,936 | 0,36 → 0,61 | 0,19 → 0,50 |
| Mahalanobis (one-class) | 0,826 | 0,879 | 0,931 | 0,38 → 0,52 | 0,00 → 0,46 |
| One-Class SVM RBF sugli stessi angoli | 0,780 | 0,842 | 0,915 | 0,10 → 0,58 | 0,00 → 0,29 |
| **One-Class SVM, kernel quantistico (7 qubit)** | 0,725 | 0,757 | 0,874 | 0,11 → 0,51 | 0,01 → 0,27 |
| One-Class SVM, kernel separabile (senza CNOT) | 0,565 | 0,625 | 0,735 | 0,01 → 0,24 | 0,00 → 0,04 |
| **VQE-inspired 7 qubit, 44 parametri (supervisionato)** | 0,606 | 0,644 | 0,652 | 0,03 → 0,11 | 0,01 → 0,02 |

Test di DeLong (60 cm, 0 dBm):
- kernel quantistico contro il miglior one-class classico: differenza da −0,06 a −0,13, p < 0,001 per ogni W;
- VQE contro Random Forest: circa −0,38, p < 0,001.

## Cosa dice
1. **Il regime è difficile ma non impossibile:** a 0 dBm e 60 cm il miglior modello passa dal 51% all'84% di rilevamento all'1% di falsi allarmi quando W va da 8 a 128 righe. Il budget di report conta molto (RQ4).
2. **Nessun vantaggio quantistico** (RQ3): il kernel quantistico è sempre sotto l'RBF calcolato sugli stessi angoli e sotto i rilevatori one-class classici. L'entanglement però aiuta rispetto al controllo separabile (0,725 contro 0,565 con 8 righe).
3. **Il VQE a 7 qubit non impara bene questo compito** (AUROC 0,61–0,65): con 600 passi e un'architettura a 2 layer resta molto sotto un MLP di dimensione simile. Una messa a punto potrebbe migliorarlo, ma il divario con il Random Forest (0,98) è ampio.
4. **Le covarianze per finestra peggiorano** i one-class rispetto alle sole medie (RQ2 negativa anche qui).
5. A 60 cm con 5 e 10 dBm, e a 40 cm, tutti i modelli classici sono già a 0,96–1,00.

Limite: le positive a 60 cm e 0 dBm vengono da 28 file. Gli intervalli di confidenza sono ampi (±0,03–0,05) e assumono finestre indipendenti, ma più finestre dello stesso file sono correlate.
