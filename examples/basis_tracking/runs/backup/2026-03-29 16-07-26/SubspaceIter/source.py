import torch


class SubspaceIter:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')
        self.B = None

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        if self.B is None:
            L, self.B = torch.linalg.eigh(self.A)

        else:
            power_iter = self.A @ self.B
            self.B, R = torch.linalg.qr(power_iter)

        return self.B
