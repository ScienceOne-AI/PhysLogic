import re
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def split_into_sentences(text: str | None) -> list[str] | None:
    """Split reasoning text into sentence-like units with the paper's light rule."""
    if text is None:
        return None
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def calculate_similarity_matrix(
    logical_nexuses: list[str],
    reasoning_sentences: list[str],
    encoder_model: Any,
) -> np.ndarray:
    """Compute M[i, j], the cosine similarity between nexus i and reasoning step j."""
    nexus_embeddings = encoder_model.encode(logical_nexuses)
    sentence_embeddings = encoder_model.encode(reasoning_sentences)
    M = cosine_similarity(nexus_embeddings, sentence_embeddings)
    return M


def greedy_matching(M: np.ndarray, threshold: float = 0.3) -> list[tuple[int, int]]:
    """Greedy one-to-one matching used by Logical Fidelity."""
    n, m = M.shape
    matched_pairs = []
    used_nexuses = set()
    used_sentences = set()
    M_copy = M.copy()
    while True:
        max_sim = -1
        max_i, max_j = -1, -1
        for i in range(n):
            if i in used_nexuses:
                continue
            for j in range(m):
                if j in used_sentences:
                    continue
                if M_copy[i, j] > max_sim and M_copy[i, j] >= threshold:
                    max_sim = M_copy[i, j]
                    max_i, max_j = i, j
        if max_sim == -1:
            break
        matched_pairs.append((max_i, max_j))
        used_nexuses.add(max_i)
        used_sentences.add(max_j)
        M_copy[max_i, :] = -1
        M_copy[:, max_j] = -1
    return matched_pairs


def calculate_f_score(
    logical_nexuses: list[str],
    logical_nexus_weights: list[float],
    reasoning_sentences: list[str],
    M: np.ndarray,
    threshold: float = 0.3,
) -> tuple[float, float, float]:
    """
    Logical Fidelity.

    Recall is the weighted coverage of ground-truth logical nexuses. Precision is
    the fraction of reasoning sentences matched to a nexus. F is their harmonic
    mean, following the paper.
    """
    _, m = M.shape
    matched_pairs = greedy_matching(M, threshold)
    total_weight = sum(logical_nexus_weights)
    weighted_similarities = 0.0
    for i, j in matched_pairs:
        weighted_similarities += logical_nexus_weights[i] * M[i, j]
    recall = weighted_similarities / total_weight if total_weight > 0 else 0.0
    precision = len(matched_pairs) / m if m > 0 else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f_score, recall, precision


def calculate_o_score(
    logical_nexuses: list[str],
    logical_nexus_weights: list[float],
    reasoning_sentences: list[str],
    M: np.ndarray,
) -> float:
    """Causal Connection score based on weighted positional centroids."""
    n, m = M.shape
    positional_centroids = []
    for i in range(n):
        numerator = 0.0
        denominator = 0.0
        for j in range(m):
            numerator += (j + 1) * M[i, j]
            denominator += M[i, j]
        centroid = (numerator / denominator) if denominator > 0 else 0.0
        positional_centroids.append(centroid)
    total_weight = 0.0
    correct_weight = 0.0
    for i in range(n):
        for k in range(i + 1, n):
            pair_weight = logical_nexus_weights[i] + logical_nexus_weights[k]
            total_weight += pair_weight
            if positional_centroids[i] < positional_centroids[k]:
                correct_weight += pair_weight
    o_score = correct_weight / total_weight if total_weight > 0 else 0.5
    return o_score


def calculate_p_score(
    logical_nexuses: list[str],
    reasoning_sentences: list[str],
    M: np.ndarray,
) -> float:
    """Inferential Progress score using novelty of focus across reasoning steps."""
    _, m = M.shape
    if m <= 1:
        return 1.0
    novelty_scores = []
    for j in range(1, m):
        current_vector = M[:, j]
        max_similarity = 0.0
        for k in range(j):
            previous_vector = M[:, k]
            dot_product = np.dot(current_vector, previous_vector)
            norm_current = np.linalg.norm(current_vector)
            norm_previous = np.linalg.norm(previous_vector)
            if norm_current > 0 and norm_previous > 0:
                similarity = dot_product / (norm_current * norm_previous)
                max_similarity = max(max_similarity, similarity)
        novelty_scores.append(1 - max_similarity)
    p_score = float(np.mean(novelty_scores)) if novelty_scores else 1.0
    return p_score


def compute_logicality_metrics(
    logical_nexuses: list[str],
    logical_nexus_weights: list[float],
    reasoning: str | None,
    encoder_model: Any,
    M: np.ndarray | None = None,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Evaluate reasoning with logical nexus alignment.

    Returns Logical Fidelity (F_score), Causal Connection (O_score),
    Inferential Progress (P_score), and supporting alignment details.
    """
    if len(logical_nexuses) != len(logical_nexus_weights):
        raise ValueError("logical_nexuses and logical_nexus_weights must have the same length")
    reasoning_sentences = split_into_sentences(reasoning)
    if not reasoning_sentences:
        return {
            'F_score': 0.0,
            'O_score': 0.0,
            'P_score': 0.0,
            'M': [],
            'reasoning_sentences': [],
            'recall': 0.0,
            'precision': 0.0,
        }
    if M is None:
        M = calculate_similarity_matrix(logical_nexuses, reasoning_sentences, encoder_model)
    f_score, recall, precision = calculate_f_score(logical_nexuses, logical_nexus_weights, reasoning_sentences, M, threshold)
    o_score = calculate_o_score(logical_nexuses, logical_nexus_weights, reasoning_sentences, M)
    p_score = calculate_p_score(logical_nexuses, reasoning_sentences, M)
    return {
        'F_score': float(f_score),
        'O_score': float(o_score),
        'P_score': float(p_score),
        'similarity_matrix': M.astype(float).tolist(),
        'reasoning_sentences': reasoning_sentences,
        'recall': float(recall),
        'precision': float(precision),
    }
