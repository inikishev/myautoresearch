import torch


class BasisUpdater:
    """Maintains approximate diagonalization of EMA matrix A = A*beta + UU^T*(1-beta).

    Optimized version with adaptive PI frequency:
    - k < m: Rotation every step + PI every 6
    - k >= m, m small: Full eigh
    - k >= m, m large: PI every 7 (optimal frequency found)
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
            else:
                _, self.B = torch.linalg.eigh(self.A)
            return self.B

        if k < m:
            return self._step_small_k(k)
        elif m <= 500:
            _, self.B = torch.linalg.eigh(self.A)
            return self.B
        else:
            return self._step_large_m()

    def _step_small_k(self, k):
        """Rotation every step + PI every 6."""
        Bk = self.B[:, :k]
        C = Bk.T @ self.A @ Bk
        _, evecs = torch.linalg.eigh(C)
        self.B[:, :k] = Bk @ evecs

        if self.step_count % 6 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q

        return self.B

    def _step_large_m(self):
        """PI every 7 steps for k >= m with large m."""
        if self.step_count % 7 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q
        return self.B