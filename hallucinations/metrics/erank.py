"""
This module implements the Diff-eRank metric which evaluates LLMs by examining their hidden
representations to quantify how models discard redundant information after training.
The metric is rooted in information theory and geometry principles and can be used for both
single-modal (language) and multi-modal settings.

Reference:
    Wei, L., Tan, Z., Li, C., Wang, J., & Huang, W. (2024).
    Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models.
    NeurIPS 2024.

Adapted from https://github.com/waltonfuture/Diff-eRank
"""

import math

import torch
from torch import Tensor

from hallucinations.metrics.common import normalize


@torch.inference_mode()
def compute_erank(R: Tensor) -> float:
    """Computes effective rank on input representation tensor using entropy.

    The effective rank is calculated by computing the entropy of the normalized singular values
    of the covariance matrix. This quantifies the number of significant dimensions in the
    representation space, with lower values indicating more compressed representation.
    """
    # Compute normalized covariance matrix
    A = torch.cov(normalize(R))

    # Get singular values
    eig_val = torch.linalg.svdvals(A / torch.trace(A))

    # Compute effective rank
    erank = -(eig_val * torch.log(eig_val)).nansum().item()
    erank = math.exp(erank)
    return erank
