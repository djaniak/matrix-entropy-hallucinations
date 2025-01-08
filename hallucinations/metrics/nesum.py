"""
This module implements the Normalized Effective Sum (NE-Sum) metric which evaluates the
information content of neural network representations by measuring their effective rank.
The metric helps quantify feature collapse and whitening in self-supervised learning.

Reference:
    He, B., & Ozay, M. (2022). Exploring the Gap between Collapsed & Whitened Features
    in Self-Supervised Learning. Proceedings of the 39th International Conference on
    Machine Learning, in Proceedings of Machine Learning Research, 162:8613-8634.
    https://proceedings.mlr.press/v162/he22c.html
"""

import torch
from torch import Tensor

from hallucinations.metrics.common import normalize


@torch.inference_mode()
def compute_nesum(R: Tensor) -> float:
    """Computes normalized effective sum (NE-Sum) on input representation tensor.

    NE-Sum is calculated as the sum of normalized singular values divided by the largest singular value.
    This quantifies the effective rank of the representation's covariance matrix.
    """
    # Compute normalized covariance matrix
    A = torch.cov(normalize(R))

    # Get singular values and normalize
    eig_val = torch.linalg.svdvals(A / torch.trace(A))
    normalized_eig_val = eig_val / torch.sum(eig_val + 1e-8)

    # Compute NE-Sum
    nesum = torch.sum(normalized_eig_val / normalized_eig_val[0]).item()
    return nesum
