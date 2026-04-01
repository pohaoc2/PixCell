import numpy as np

from tools.uni_cosine_similarity import cosine_similarity_uni, flatten_uni_npy


def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity_uni(v, v) - 1.0) < 1e-7


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity_uni(a, b)) < 1e-9


def test_flatten_uni_npy():
    x = np.ones((1, 1, 1, 8))
    assert flatten_uni_npy(x).shape == (8,)


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity_uni(np.ones(3), np.ones(4))
