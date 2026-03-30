import torch


class BasisUpdater:
    """Pure subspace iteration from random orthogonal B. Freq=3."""

    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device="cuda")
        # Random orthogonal initialization (better than I for convergence)
        B = torch.randn(m, m, device="cuda")
        self.B, _ = torch.linalg.qr(B)
        self.current_step = 0
        self.update_freq = 3

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)

        if self.current_step % self.update_freq == 0:
            self.B, R = torch.linalg.qr(self.A @ self.B)

        self.current_step += 1
        return self.B
