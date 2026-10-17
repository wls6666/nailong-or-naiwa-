"""Shared real inference, metrics, and rejection semantics for evaluation/verification."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all(): raise ValueError('Non-finite model logits')
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def decisions(probabilities, mapping, threshold):
    probabilities = np.asarray(probabilities)
    targets = [mapping['nailong'], mapping['naiwa']]
    raw = probabilities.argmax(axis=1)
    max_target = probabilities[:, targets].max(axis=1)
    rejected = (raw != mapping['other']) & (max_target < threshold)
    result = raw.copy(); result[rejected] = mapping['other']
    return result, rejected, max_target


def metrics(labels, predictions, mapping):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    order = sorted(mapping.values())
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=order, zero_division=0)
    other = labels == mapping['other']; targets = ~other
    report = {'accuracy': float(accuracy_score(labels, predictions)),
              'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
              'per_class': {name: {'precision': float(precision[idx]), 'recall': float(recall[idx]),
                                   'f1': float(f1[idx]), 'support': int(support[idx])} for name, idx in mapping.items()},
              'confusion_matrix': confusion_matrix(labels, predictions, labels=order).tolist(),
              'confusion_matrix_axes': 'rows=true; columns=prediction; index order',
              'target_accuracy': float(np.mean(labels[targets] == predictions[targets])),
              'other_false_accept_rate': float(np.mean(predictions[other] != mapping['other'])),
              'other_false_accept_count': int(np.sum(predictions[other] != mapping['other']))}
    return report


def infer_dataset(model, dataset):
    labels, logits = [], []
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=32, shuffle=False):
            labels.extend(y.tolist()); logits.extend(model(x).tolist())
    return np.asarray(labels), np.asarray(logits, dtype=np.float32)
