import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
from typing import List, Union


def split_text_word(lines: Union[str, List[str]]) -> List[List[str]]:

    if isinstance(lines, str): # Check if input is a file path
        if not os.path.exists(lines):
            print(f"Warning: File not found at {lines}. Returning empty list.")
            return []
        try:
            with open(lines, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {lines}: {e}. Returning empty list.")
            return []
    # Process list of strings (or lines read from file)
    return [line.strip().split() for line in lines if line.strip()]


def load_labels_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"Error: Label file not found at {path}. Returning empty array.")
        return np.array([])
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
        return np.array(labels)
    except Exception as e:
        print(f"Error reading or parsing label file {path}: {e}. Returning empty array.")
        return np.array([])
def _cls(train_theta, test_theta, train_labels, test_labels, gamma='scale'):
    if len(np.unique(train_labels)) <= 1:
        print("Warning: Only one class present in training data. Classifier might not train meaningfully.")
        # Return default/failure metrics if training is not possible/meaningful
        return {'acc': 0.0, 'macro-F1': 0.0}

    try:
        clf = SVC(gamma=gamma, probability=False) # probability=True is slower if not needed
        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        return {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro', zero_division=0) # Handle zero division
        }
    except Exception as e:
        print(f"Error during SVM classification: {e}")
        return {'acc': np.nan, 'macro-F1': np.nan}


def crosslingual_cls(train_theta_en, train_theta_cn,
                     test_theta_en, test_theta_cn,
                     train_labels_en, train_labels_cn,
                     test_labels_en, test_labels_cn):
    results = {
        'intra_en': _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en),
        'intra_cn': _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn),
        'cross_en': _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en), # Train CN, Test EN
        'cross_cn': _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn), # Train EN, Test CN
    }
    return results


def print_results(results):
    """
    Print classification results in a formatted way.

    Args:
        results: Dictionary with classification metrics (output from crosslingual_cls)
    """
    for key, val in results.items():
        print(f"\n>>> {key.upper()}")
        # Check if metrics are valid numbers before formatting
        acc_str = f"{val.get('acc', 'N/A'):.4f}" if isinstance(val.get('acc'), (int, float)) and not np.isnan(val.get('acc')) else "N/A"
        f1_str = f"{val.get('macro-F1', 'N/A'):.4f}" if isinstance(val.get('macro-F1'), (int, float)) and not np.isnan(val.get('macro-F1')) else "N/A"
        print(f"  Accuracy   : {acc_str}")
        print(f"  Macro-F1   : {f1_str}")

