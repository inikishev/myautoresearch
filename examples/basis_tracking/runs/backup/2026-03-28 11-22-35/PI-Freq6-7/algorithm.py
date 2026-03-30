import torch


class BasisUpdater:
    """Maintains approximate diagonalization of EMA matrix A = A*beta + UU^T*(1-beta).

    Optimizes PI frequency based on the rate of change of A.
    Key insight: A changes slowly when beta is close to 1.
    We can do PI less frequently and still maintain good diagonalization.
    """

    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device="cuda")
        self.m = m
        self.B = None
        self.step_count = 0
        # Use different frequencies for k<m vs k>=m
        self.pi_freq_small_k = 6  # For k < m
        self.pi_freq_large_m = 7  # For k >= m

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
        """For k < m: rotation + PI every pi_freq_small_k steps."""
        # Rotation step (cheap, O(k^3) for eigh on k×k matrix)
        Bk = self.B[:, :k]
        C = Bk.T @ self.A @ Bk
        C = C + 1e-8 * torch.eye(k, device=C.device, dtype=C.dtype)
        _, evecs = torch.linalg.eigh(C)
        self.B[:, :k] = Bk @ evecs

        # PI for k < m
        if self.step_count % self.pi_freq_small_k == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q

        return self.B

    def _step_large_m(self):
        """For k >= m with large m: PI every pi_freq_large_m steps."""
        if self.step_count % self.pi_freq_large_m == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q
        return self.B
