"""
This module implements core quantities for Matrix Renyi's alpha entropy and divergence.

Adapted from https://github.com/uk-cliplab/representation-itl
"""

from functools import reduce
from typing import List, Literal, Optional

import torch
from torch import Tensor

from hallucinations.metrics.common import normalize


def generalized_information_potential(
    K: Tensor, alpha: float, allow_frobenius_speedup: bool = True
) -> Tensor:
    """Compute generalized information potential of order alpha.

    GIP_alpha(K) = trace(K_^alpha), where K^alpha is matrix raised to alpha power
    and K_ = K/trace(K) is normalized so trace(K_) = 1.

    Args:
        K: (N x N) Gram matrix
        alpha: Order of entropy
        allow_frobenius_speedup: Whether to use faster Frobenius norm for alpha=2

    Returns:
        Generalized information potential of order alpha
    """
    if allow_frobenius_speedup and alpha == 2:
        return frobenius_gip(K)

    ek, _ = torch.linalg.eigh(K)
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]
    mek = mek / torch.sum(mek)
    return torch.sum(torch.exp(alpha * torch.log(mek)))


def frobenius_gip(K: Tensor) -> Tensor:
    """Calculate entropy using Frobenius norm.

    Equivalent to generalized_information_potential(K, alpha=2) but faster.

    Args:
        K: (N x N) Gram matrix

    Returns:
        Generalized information potential using Frobenius norm
    """
    GIP = torch.sum(torch.pow(K, 2))
    GIP /= K.shape[0] ** 2  # Normalize so sum of eigenvalues is 1
    return GIP


def matrix_alpha_entropy(K: Tensor, alpha: float) -> Tensor:
    """Compute matrix-based alpha-entropy from spectrum of K.

    H_alpha(A) = (1/(1-alpha))log(trace(A^alpha))
    where A^alpha is matrix power of alpha (A is normalized)

    Args:
        K: (N x N) Gram matrix
        alpha: Order of entropy

    Returns:
        Alpha entropy value
    """
    if alpha == 1:  # Handle limit case
        return von_neumann_entropy(K)

    GIP = generalized_information_potential(K, alpha).real
    return (1.0 / (1.0 - alpha)) * torch.log(GIP)


def compute_matrix_alpha_entropy(R: Tensor, alpha: float) -> float:
    "Compute matrix-based alpha entropy for a given representation tensor."
    R = normalize(R)
    K = torch.cov(R)
    K = K / torch.trace(K)
    return matrix_alpha_entropy(K, alpha).item()


def matrix_alpha_joint_entropy(K_list: List[Tensor], alpha: float) -> Tensor:
    """Compute matrix-based alpha joint-entropy from spectrum of K.

    H_alpha(K) = (1/(1-alpha))log(trace(K^alpha))
    where K^alpha is matrix power of K (K is normalized)

    Args:
        K_list: List of (N x N) Gram matrices
        alpha: Order of entropy

    Returns:
        Alpha joint entropy value
    """
    K = reduce(lambda x, y: x * y, K_list)
    return matrix_alpha_entropy(K, alpha)


def matrix_alpha_conditional_entropy(Kx: Tensor, Ky: Tensor, alpha: float) -> Tensor:
    """Compute matrix-based alpha conditional entropy."""
    Kxy = Kx * Ky
    Hxy = matrix_alpha_entropy(Kxy, alpha=alpha)
    Hy = matrix_alpha_entropy(Ky, alpha=alpha)
    return Hxy - Hy


def matrix_alpha_mutual_information(Kx: Tensor, Ky: Tensor, alpha: float) -> Tensor:
    """Compute matrix-based alpha mutual information."""
    Kxy = Kx * Ky
    Hxy = matrix_alpha_entropy(Kxy, alpha=alpha)
    Hx = matrix_alpha_entropy(Kx, alpha=alpha)
    Hy = matrix_alpha_entropy(Ky, alpha=alpha)
    return Hx + Hy - Hxy


def schatten_norm(T: Tensor, p: float) -> Tensor:
    """Compute p-th power of Schatten p-norm of matrix T.

    Args:
        T: Input matrix
        p: Order of norm

    Returns:
        (||T||_p)^p
    """
    _, s, _ = torch.svd(T, some=True)
    return torch.sum(s.pow(p))


def normalize_triplet(Kx: Tensor, Ky: Tensor, Kxy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize a triplet of kernel matrices."""
    Kxy = Kxy / torch.sqrt(torch.trace(Kx) * torch.trace(Ky))
    Kx = Kx / torch.trace(Kx)
    Ky = Ky / torch.trace(Ky)
    return Kx, Ky, Kxy


def schatten_divergence(
    Kx: Tensor, Ky: Tensor, Kxy: Tensor, p: float, normalize: bool = False
) -> None:
    pass


def schatten1_divergence(Kx: Tensor, Ky: Tensor, Kxy: Tensor, normalize: bool = False) -> Tensor:
    """Compute Schatten-1 divergence."""
    if normalize:
        Kx, Ky, Kxy = normalize_triplet(Kx, Ky, Kxy)
    return 1 - schatten_norm(Kxy, p=1.0)


def matrix_alpha_divergence(
    Kx: Tensor, Ky: Tensor, Kxy: Tensor, alpha: float, normalize: bool = False
) -> Tensor:
    """Compute matrix-based alpha-divergence.

    H_alpha(K) = (1/(1-alpha))log(trace(K^alpha))
    where K^alpha is matrix power of K (K is normalized)

    Args:
        Kx: (N x N) First Gram matrix
        Ky: (N x N) Second Gram matrix
        Kxy: (N x N) Joint Gram matrix
        alpha: Order of divergence
        normalize: Whether to normalize matrices

    Returns:
        Alpha divergence based on spectra of Kx, Ky, Kxy
    """
    if normalize:
        Kx, Ky, Kxy = normalize_triplet(Kx, Ky, Kxy)

    ex, vx = torch.linalg.eigh(Kx)
    ey, vy = torch.linalg.eigh(Ky)

    # Handle negative eigenvalues
    mx = torch.gt(ex, 0.0)
    my = torch.gt(ey, 0.0)
    mex = ex[mx] / torch.sum(ex[mx])
    mey = ey[my] / torch.sum(ey[my])
    mvx = vx[:, mx]
    mvy = vy[:, my]

    M = torch.square(torch.matmul(mvx.t(), torch.matmul(Kxy, mvy)))
    B = torch.matmul(
        torch.pow(mex, (alpha - 1))[None, :], torch.matmul(M, torch.pow(mey, -alpha)[:, None])
    )
    return torch.log(B) / (alpha - 1)


def von_neumann_entropy(K: Tensor, low_rank: bool = False, rank: Optional[int] = None) -> Tensor:
    """Compute von Neumann entropy of matrix K.

    Args:
        K: Input matrix
        low_rank: Whether to use low-rank approximation
        rank: Rank to use for low-rank approximation

    Returns:
        von Neumann entropy value
    """
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)

    if low_rank:
        assert rank is not None, "Rank must be provided for low-rank approximation"
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        remainder = ek.sum() - ek_lr.sum()
        ek_lr[: (n - rank)] = remainder / (n - rank)
        mk = torch.gt(ek_lr, 0.0)
        mek = ek_lr[mk]
    else:
        mk = torch.gt(ek, 0.0)
        mek = ek[mk]

    mek = mek / mek.sum()
    return -1 * torch.sum(mek * torch.log(mek))


def von_neumann_eigen_values(
    Ev: Tensor, low_rank: bool = False, rank_proportion: float = 0.9
) -> Tensor:
    """Compute von Neumann entropy from eigenvalues.

    Args:
        Ev: Eigenvalues (assumed to be ordered descendingly)
        low_rank: Whether to use low-rank approximation
        rank_proportion: Proportion of eigenvalues to keep if using low-rank

    Returns:
        von Neumann entropy value
    """
    if low_rank:
        n_eig = int(rank_proportion * Ev.shape[0])
        Ev_lr = torch.zeros_like(Ev)
        Ev_lr[:n_eig] = Ev[:n_eig]
        Ev_lr[n_eig:] = torch.mean(Ev[n_eig:])
        Ev_lr = Ev_lr / torch.sum(Ev_lr)
        return -1 * torch.sum(Ev_lr * torch.log(Ev_lr))

    mk = torch.gt(Ev, 0.0)
    mek = Ev[mk] / torch.sum(Ev[mk])
    return -1 * torch.sum(mek * torch.log(mek))


def row_wise_kronecker_product(X: Tensor, Y: Tensor) -> Tensor:
    """Compute row-wise Kronecker product of matrices.

    Args:
        X: First matrix
        Y: Second matrix

    Returns:
        Row-wise Kronecker product
    """
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Both arrays must be 2-dimensional")
    if not X.shape[0] == Y.shape[0]:
        raise ValueError("Number of rows must match")

    X = X.T
    Y = Y.T
    c = X[..., :, None, :] * Y[..., None, :, :]
    transposed_kr = c.reshape((-1,) + c.shape[2:])
    return transposed_kr.T


def rep_mutual_information(
    X: Tensor, Y: Tensor, type: Literal["covariance", "kernel"] = "covariance"
) -> Tensor:
    """Compute representational mutual information.

    Args:
        X: First representation matrix
        Y: Second representation matrix
        type: Type of computation ("covariance" or "kernel")

    Returns:
        Mutual information value
    """
    n = X.shape[0]
    phi_xy = row_wise_kronecker_product(X, Y)

    if type == "covariance":
        cov_x = (1 / n) * X.T @ X
        cov_y = (1 / n) * Y.T @ Y
        cov_xy = (1 / n) * phi_xy.T @ phi_xy
        Hx = von_neumann_entropy(cov_x)
        Hy = von_neumann_entropy(cov_y)
        Hxy = von_neumann_entropy(cov_xy)
    else:  # kernel
        Kx = (1 / n) * X @ X.T
        Ky = (1 / n) * Y @ Y.T
        Kxy = (1 / n) * phi_xy @ phi_xy.T
        Hx = von_neumann_entropy(Kx)
        Hy = von_neumann_entropy(Ky)
        Hxy = von_neumann_entropy(Kxy)

    return Hx + Hy - Hxy
