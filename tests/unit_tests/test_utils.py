"""Unit tests for the vendored vector-math utilities."""

import numpy as np

from langchain_falkordb.utils import DistanceStrategy, maximal_marginal_relevance


def test_distance_strategy_members() -> None:
    assert DistanceStrategy.COSINE == "COSINE"
    assert DistanceStrategy.EUCLIDEAN_DISTANCE == "EUCLIDEAN_DISTANCE"


def test_maximal_marginal_relevance_prefers_diversity() -> None:
    query = np.array([1.0, 0.0])
    # First two vectors are nearly identical; third is orthogonal.
    embeddings = [[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]
    selected = maximal_marginal_relevance(query, embeddings, lambda_mult=0.4, k=2)
    assert selected == [0, 2]


def test_maximal_marginal_relevance_k_larger_than_input() -> None:
    query = np.array([1.0, 0.0])
    selected = maximal_marginal_relevance(query, [[1.0, 0.0]], k=5)
    assert selected == [0]

    assert maximal_marginal_relevance(query, [], k=3) == []
