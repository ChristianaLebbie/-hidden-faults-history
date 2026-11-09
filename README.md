# hidden-faults-history

Code for the paper "Hidden Faults Under Partial Sensor Acquisition: Revisiting the AI4I-PMDI Benchmark and the Value of Machine History" by Christiana Lebbie (Huazhong University of Science and Technology).

## What is here
- `src/pfi/`: data loading, inspection schedules, history features, models, GRU-D and evaluation
- `experiments/`: one script per experiment (exp1 to exp18)
- `analysis/`: summaries, statistics and figures
- `tests/test_no_future_leak.py`: checks that history features use only earlier inspections of the same machine

## Data (not included)
- AI4I 2020 (Matzka, 2020) and AI4I-PMDI (Autran et al., 2024)
- C-MAPSS FD001 (NASA Prognostics Data Repository)
- MetroPT-3 (UCI Machine Learning Repository)

Download them and place them in a `data/` folder next to `src/`.

## Requirements
Python 3.13, XGBoost 3.4.1, scikit-learn 1.9.1, pandas 3.0.6, PyTorch (for GRU-D). Install with:

    pip install -r requirements.txt

## Run
    python experiments/exp1_history_methods.py
    python tests/test_no_future_leak.py
