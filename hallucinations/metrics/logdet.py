"""
This module implements the hidden score metric from the paper "LLM-Check: Investigating Detection of Hallucinations in Large Language Models".
The hidden score is computed as the log determinant of the covariance matrix of the hidden states, which quantifies
the amount of information contained in the model's internal representations.

Reference:
    Sriramanan, G., Bharti, S., Sadasivan, V. S., Saha, S., Kattakinda, P., & Feizi, S. (2024).
    LLM-Check: Investigating Detection of Hallucinations in Large Language Models.
    NeurIPS 2024.
    https://openreview.net/forum?id=LYx4w3CAgy

Adapted from https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection
"""

import torch

from hallucinations.metrics.common import normalize


@torch.inference_mode()
def compute_logdet(Z: torch.Tensor, alpha: float = 0.001) -> float:
    """Compute the mean log singular value of a centered covariance matrix.

    This function centers the data and computes the singular value decomposition
    (SVD) of the resulting covariance matrix. It then returns the mean of the
    log singular values, regularized by `alpha`.
    """
    Z = torch.transpose(Z, 0, 1)
    n = Z.shape[0]

    # Compute centering matrix J
    J = torch.eye(n, device=Z.device) - torch.ones(n, n, device=Z.device) / n

    # Compute centered covariance matrix
    Z_t = Z.t()
    Sigma = Z_t @ J @ Z

    # Add regularization
    Sigma += alpha * torch.eye(Sigma.shape[0], device=Z.device)

    # Compute SVD and mean log singular values
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean().item()

    return eigscore


@torch.inference_mode()
def compute_logdet_ver1(Z: torch.Tensor) -> float:
    """Compute the mean log singular value of a centered covariance matrix.

    This function centers the data and computes the singular value decomposition
    (SVD) of the resulting covariance matrix. It then returns the mean of the
    log singular values, regularized by `alpha`.
    """
    # Compute normalized covariance matrix
    A = torch.cov(normalize(Z))

    # Compute SVD and mean log singular values
    eigvals = torch.linalg.svdvals(A / torch.trace(A))
    eigscore = torch.log(eigvals).mean().item()

    return eigscore


# @torch.inference_mode()
# def compute_logdet_ver2(Z: torch.Tensor, alpha: float = 0.001) -> float:
#     """Compute the mean log singular value of a centered covariance matrix.

#     This function centers the data and computes the singular value decomposition
#     (SVD) of the resulting covariance matrix. It then returns the mean of the
#     log singular values, regularized by `alpha`.
#     """
#     # Compute normalized covariance matrix
#     A = torch.cov(normalize(Z))
#     A += alpha * torch.eye(A.shape[0], device=Z.device)

#     # Compute SVD and mean log singular values
#     eigvals = torch.linalg.svdvals(A)
#     eigscore = torch.log(eigvals).mean().item()

#     return eigscore


# @torch.inference_mode()
# def compute_logdet_ver3(Z: torch.Tensor, alpha: float = 0.001) -> float:
#     """Compute the mean log singular value of a centered covariance matrix.

#     This function centers the data and computes the singular value decomposition
#     (SVD) of the resulting covariance matrix. It then returns the mean of the
#     log singular values, regularized by `alpha`.
#     """
#     Z = torch.transpose(Z, 0, 1)

#     # Compute normalized covariance matrix
#     A = torch.cov(normalize(Z))

#     # Add regularization
#     A += alpha * torch.eye(A.shape[0], device=Z.device)

#     # Compute SVD and mean log singular values
#     eigvals = torch.linalg.svdvals(A)
#     eigscore = torch.log(eigvals).mean().item()

#     return eigscore
