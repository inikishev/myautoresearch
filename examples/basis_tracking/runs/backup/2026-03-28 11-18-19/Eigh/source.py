import torch

class Eigh:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        L, Q = torch.linalg.eigh(self.A)
        return Q