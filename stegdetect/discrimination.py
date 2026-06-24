"""
Fisher Linear Discriminant Analysis for steganography detection.

Implements the classification-based detector from the original stegdetect.
Uses feature extraction transforms to build a classifier.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingSample:
    """A training sample for the classifier."""
    filename: str
    positive: bool  # True = contains steganography
    features: np.ndarray  # feature vector


@dataclass
class Classifier:
    """Fisher LDA classifier."""
    name: str = ""
    b: np.ndarray = field(default_factory=lambda: np.array([]))  # projection vector
    k: float = 0.0  # decision boundary
    proj_pos: float = 0.0  # positive sample projection
    proj_neg: float = 0.0  # negative sample projection
    transform_name: str = ""


def fisher_lda_train(positive_features: list[np.ndarray],
                     negative_features: list[np.ndarray]) -> Optional[Classifier]:
    """Train a Fisher LDA classifier.

    Algorithm:
    1. Compute mean vectors for positive and negative samples
    2. Compute pooled covariance matrix
    3. Compute projection vector: b = cov^(-1) * (mean_pos - mean_neg)
    4. Compute decision boundary: k = (proj_pos + proj_neg) / 2

    Args:
        positive_features: list of feature vectors from stego images
        negative_features: list of feature vectors from clean images

    Returns:
        Classifier object, or None if training fails
    """
    if not positive_features or not negative_features:
        return None

    n_pos = len(positive_features)
    n_neg = len(negative_features)
    n_features = len(positive_features[0])

    # Check all feature vectors have same length
    for f in positive_features + negative_features:
        if len(f) != n_features:
            return None

    # Compute means
    mean_pos = np.mean(positive_features, axis=0)
    mean_neg = np.mean(negative_features, axis=0)

    # Compute pooled covariance
    cov_pos = np.cov(np.array(positive_features).T, bias=False) if n_pos > 1 else np.zeros((n_features, n_features))
    cov_neg = np.cov(np.array(negative_features).T, bias=False) if n_neg > 1 else np.zeros((n_features, n_features))

    # Pooled covariance
    pooled_cov = ((n_pos - 1) * cov_pos + (n_neg - 1) * cov_neg) / (n_pos + n_neg - 2)

    # Add regularization for numerical stability
    pooled_cov += np.eye(n_features) * 1e-6

    try:
        pooled_cov_inv = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        return None

    # Projection vector
    diff = mean_pos - mean_neg
    b = pooled_cov_inv @ diff

    # Normalize
    b_norm = np.linalg.norm(b)
    if b_norm > 0:
        b = b / b_norm

    # Project means
    proj_pos = float(np.dot(b, mean_pos))
    proj_neg = float(np.dot(b, mean_neg))

    # Decision boundary
    k = (proj_pos + proj_neg) / 2.0

    classifier = Classifier()
    classifier.b = b
    classifier.k = k
    classifier.proj_pos = proj_pos
    classifier.proj_neg = proj_neg

    return classifier


def fisher_lda_classify(classifier: Classifier, features: np.ndarray) -> tuple[bool, float]:
    """Classify a sample using the trained Fisher LDA classifier.

    Args:
        classifier: trained classifier
        features: feature vector

    Returns:
        (is_stego, confidence) tuple
    """
    proj = float(np.dot(classifier.b, features))

    # Distance from decision boundary
    dist = proj - classifier.k
    total_dist = classifier.proj_pos - classifier.proj_neg
    if total_dist == 0:
        return False, 0.0

    confidence = dist / total_dist
    is_stego = proj > classifier.k

    return is_stego, confidence


def train_and_test(positive_files: list[str], negative_files: list[str],
                   extract_fn, test_ratio: float = 0.2) -> Optional[Classifier]:
    """Train a classifier with train/test split.

    Args:
        positive_files: list of stego image file paths
        negative_files: list of clean image file paths
        extract_fn: function to extract features from a file
        test_ratio: fraction of data to use for testing

    Returns:
        Trained classifier, or None if training fails
    """
    import random

    # Extract features
    pos_features = []
    for f in positive_files:
        try:
            feat = extract_fn(f)
            if feat is not None:
                pos_features.append(feat)
        except Exception:
            continue

    neg_features = []
    for f in negative_files:
        try:
            feat = extract_fn(f)
            if feat is not None:
                neg_features.append(feat)
        except Exception:
            continue

    if not pos_features or not neg_features:
        return None

    # Split
    random.shuffle(pos_features)
    random.shuffle(neg_features)

    n_pos_test = max(1, int(len(pos_features) * test_ratio))
    n_neg_test = max(1, int(len(neg_features) * test_ratio))

    train_pos = pos_features[n_pos_test:]
    test_pos = pos_features[:n_pos_test]
    train_neg = neg_features[n_neg_test:]
    test_neg = neg_features[:n_neg_test]

    # Train
    classifier = fisher_lda_train(train_pos, train_neg)
    if classifier is None:
        return None

    # Test
    correct = 0
    total = len(test_pos) + len(test_neg)

    for feat in test_pos:
        is_stego, _ = fisher_lda_classify(classifier, feat)
        if is_stego:
            correct += 1

    for feat in test_neg:
        is_stego, _ = fisher_lda_classify(classifier, feat)
        if not is_stego:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Classifier accuracy: {accuracy:.2%} ({correct}/{total})")

    return classifier
