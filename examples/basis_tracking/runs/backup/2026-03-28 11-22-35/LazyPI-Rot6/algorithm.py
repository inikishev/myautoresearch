import torch


class BasisUpdater:
    """Maintains approximate diagonalization of EMA matrix A = A*beta + UU^T*(1-beta).

    Strategy:
    - k < m: Lazy full power iteration (A @ B + QR every K steps).
      Rotation within the k-dim subspace between PI steps to maintain quality.
    - k >= m, m <= 500: Full eigendecomposition each step.
    - k >= m, m > 500: Lazy full power iteration (A @ B + QR every K steps).
    """

    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device="cuda")
        self.m = m
        self.B = None
        self.step_count = 0

    def step(self, U: torch.Tensor) -> torch.Tensor:
        k = U.shape[1]
        m = self.m

        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        self.step_count += 1

        if self.B is None:
            if k < m:
                _, self.B = torch.linalg.eigh(UUt)
            elif m <= 500:
                _, self.B = torch.linalg.eigh(self.A)
            else:
                _, self.B = torch.linalg.eigh(self.A)
            return self.B

        if k < m:
            return self._step_small_k(k)
        elif m <= 500:
            _, self.B = torch.linalg.eigh(self.A)
            return self.B
        else:
            return self._step_lazy_pi()

    def _step_small_k(self, k):
        """Lazy full PI with rotation between steps for k < m."""
        # Cheap rotation within current subspace to maintain diagonalization quality
        Bk = self.B[:, :k]
        C = Bk.T @ self.A @ Bk
        _, evecs = torch.linalg.eigh(C)
        self.B[:, :k] = Bk @ evecs

        # Periodic full power iteration to expand subspace and re-orthogonalize
        if self.step_count % 6 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q

        return self.B

    def _step_lazy_pi(self):
        """Lazy full power iteration for k >= m."""
        if self.step_count % 6 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q
        return self.B
