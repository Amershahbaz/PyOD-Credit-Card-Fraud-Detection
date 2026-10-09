"""Hands-On Assignment 4: unsupervised credit-card fraud detection with PyOD."""
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('creditcard.csv'))
    parser.add_argument('--train-limit', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f'Dataset not found: {args.data}')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    df = pd.read_csv(args.data)
    expected = {'Time', 'Amount', 'Class'} | {f'V{i}' for i in range(1, 29)}
    if not expected.issubset(df.columns):
        raise ValueError('Unexpected dataset columns')
    if df.isna().any().any():
        raise ValueError('Missing values found; review data before training')
    X = df.drop(columns='Class')
    y = df['Class'].astype(int)
    # Labels are used ONLY for evaluation and stratification, never as model inputs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Limit training size for a reproducible CPU-friendly experiment.
    if args.train_limit and len(X_train_scaled) > args.train_limit:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(len(X_train_scaled), args.train_limit, replace=False)
        X_fit = X_train_scaled[chosen]
    else:
        X_fit = X_train_scaled
    # Estimated outlier proportion controls the decision threshold.
    contamination = float(y_train.mean())
    model = AutoEncoder(
        contamination=contamination,
        epoch_num=args.epochs,
        batch_size=256,
        lr=0.001,
        hidden_neuron_list=[32, 16, 8, 16, 32],
        preprocessing=False,
        verbose=1,
        random_state=SEED,
    )
    model.fit(X_fit)
    scores = model.decision_function(X_test_scaled)
    predictions = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        'dataset_rows': int(len(df)),
        'fraud_rows': int(y.sum()),
        'test_rows': int(len(y_test)),
        'training_rows_used': int(len(X_fit)),
        'fraud_prevalence': float(y.mean()),
        'roc_auc': float(roc_auc_score(y_test, scores)),
        'average_precision': float(average_precision_score(y_test, scores)),
        'confusion_matrix': cm.tolist(),
    }
    print('\nFRAUD DETECTION EXPERIMENT RESULTS')
    print(json.dumps(metrics, indent=2))
    print('\nClassification report:\n', classification_report(
        y_test, predictions, digits=4, zero_division=0))
    Path('results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_test.to_numpy() == 0], bins=60, alpha=0.65, label='Legitimate')
    plt.hist(scores[y_test.to_numpy() == 1], bins=60, alpha=0.65, label='Fraud')
    plt.xlabel('Reconstruction-based anomaly score')
    plt.ylabel('Number of transactions')
    plt.title('PyOD AutoEncoder anomaly scores on held-out transactions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('anomaly_scores.png', dpi=160)
    print('\nSaved results.json and anomaly_scores.png')


if __name__ == '__main__':
    main()
