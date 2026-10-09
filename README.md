# Fraud Detection with PyOD AutoEncoder

Unsupervised anomaly detection on the anonymized credit-card dataset.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install packages: `pip install -r requirements.txt`.
3. Place `creditcard.csv` beside `fraud_detection.py` (do not commit the dataset).
4. Run: `python fraud_detection.py --data creditcard.csv`.
5. Review `results.json` and `anomaly_scores.png`. Capture a screenshot of the terminal output for the assignment document.

The model does not use fraud labels as training features. Labels are reserved for stratified splitting, threshold prevalence estimation, and final evaluation. PyOD versions may change supported AutoEncoder constructor parameters; this example targets the PyOD 2.x interface.
