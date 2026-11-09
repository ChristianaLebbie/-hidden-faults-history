# Supervisor feedback, 26 Oct 2026 (Prof. Jing Xu) and work plan

## Feedback (summary)
Positive: clearer research question, more systematic experiments; added baselines, robustness tests and real-data analysis address earlier concerns; appreciates reporting where history or the reliability gate does not help. Keep the current chapter structure.

1. Contribution vs closest literature: compare with a sequence model for missing observations (e.g. GRU-D); check whether the age feature gives a consistent benefit; for AI4I-PMDI, separate the original authors' stated design choices from the limitations shown by my experiments.
2. Hidden-fault evaluation: define how hidden faults and their evaluation samples are selected; compare hidden- and visible-fault recall at a common false-alarm target; MetroPT: report the three test events (F2, F3, F4) separately with detection delays and false alarms, plus the average.
3. Chapter 2: develop the distinction between reconstructing the current reading and improving fault identification (FD002: history is a poor estimate of the current value yet helps classification), especially vs elapsed time and operating-condition changes.

## Work plan
| Task | Type | Chapter | Effort |
|---|---|---|---|
| T1 Age ablation: AMA with vs without age, 5 classifiers x AI4I/FD001/FD002 | rerun (extend exp13) | 1 | small |
| T2 Recall of hidden vs visible faults at a common false-alarm target (e.g. 1 per 100 normal records) + written definition of hidden faults and evaluation samples | analysis | 1 | small |
| T3 MetroPT per event F2/F3/F4: PR-AUC, detection delay, false alarms per day, plus average | extend exp10 | 1 | small |
| T4 Reconstruction vs identification: per bin of elapsed time and condition change (FD002, also AI4I/FD001), R2 of history vs classification gain | new analysis | 2 | medium |
| T5 GRU-D baseline on the same splits and schedules (PyTorch, CPU) | new model | 1 | largest |
| T6 AI4I-PMDI: authors' stated design vs limitations my experiments show | writing (needs original source) | 1 | writing |

Order: T1, T2, T3 (quick), then T4, then T5; T6 written alongside. Report whatever the evidence supports, including if GRU-D matches or beats AMA.

Known facts for T6 (from my audit): every deterministic failure in AI4I-PMDI is recorded under the control that measures its causal sensors (HDF->A, PWF->B, OSF/TWF->C), so the published schedule never hides a fault; S0 0.878 vs S2 0.401 shows the effect.
