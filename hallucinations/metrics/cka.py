# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

import math
from typing import Optional

import numpy as np
import torch


class CenteredKernelAlignment:
    """Centered Kernel Alignment (CKA) implementation using numpy arrays.

    CKA is a similarity index between two sets of features/representations that is invariant to orthogonal transformation
    and isotropic scaling. It can be used to compare neural network representations across layers and architectures.
    """

    def centering(self, K: np.ndarray) -> np.ndarray:
        """Center a kernel matrix K using the centering matrix H = I - 1/n.

        Args:
            K: Kernel matrix of shape (n, n)

        Returns:
            Centered kernel matrix of shape (n, n)
        """
        n = K.shape[0]
        unit = np.ones([n, n])
        identity = np.eye(n)
        H = identity - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Compute the RBF (Gaussian) kernel matrix for input X.

        Args:
            X: Input matrix of shape (n, d)
            sigma: Kernel bandwidth. If None, uses median distance heuristic

        Returns:
            RBF kernel matrix of shape (n, n)
        """
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X: np.ndarray, Y: np.ndarray, sigma: Optional[float]) -> float:
        """Compute kernel HSIC (Hilbert-Schmidt Independence Criterion) between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)
            sigma: Kernel bandwidth for RBF kernel

        Returns:
            Kernel HSIC value
        """
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear HSIC between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)

        Returns:
            Linear HSIC value
        """
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA similarity between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)

        Returns:
            Linear CKA similarity value between 0 and 1
        """
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
        """Compute kernel (RBF) CKA similarity between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)
            sigma: Kernel bandwidth for RBF kernel. If None, uses median distance heuristic

        Returns:
            Kernel CKA similarity value between 0 and 1
        """
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


class CudaCKA:
    """CUDA-accelerated implementation of Centered Kernel Alignment (CKA) using PyTorch tensors."""

    def __init__(self, device: torch.device):
        """Initialize CudaCKA.

        Args:
            device: PyTorch device to use for computations
        """
        self.device = device

    def centering(self, K: torch.Tensor) -> torch.Tensor:
        """Center a kernel matrix K using the centering matrix H = I - 1/n.

        Args:
            K: Kernel matrix of shape (n, n)

        Returns:
            Centered kernel matrix of shape (n, n)
        """
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """Compute the RBF (Gaussian) kernel matrix for input X.

        Args:
            X: Input matrix of shape (n, d)
            sigma: Kernel bandwidth. If None, uses median distance heuristic

        Returns:
            RBF kernel matrix of shape (n, n)
        """
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float]) -> torch.Tensor:
        """Compute kernel HSIC between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)
            sigma: Kernel bandwidth for RBF kernel

        Returns:
            Kernel HSIC value
        """
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear HSIC between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)

        Returns:
            Linear HSIC value
        """
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear CKA similarity between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)

        Returns:
            Linear CKA similarity value between 0 and 1
        """
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(
        self, X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float] = None
    ) -> torch.Tensor:
        """Compute kernel (RBF) CKA similarity between X and Y.

        Args:
            X: First input matrix of shape (n, d1)
            Y: Second input matrix of shape (n, d2)
            sigma: Kernel bandwidth for RBF kernel. If None, uses median distance heuristic

        Returns:
            Kernel CKA similarity value between 0 and 1
        """
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
