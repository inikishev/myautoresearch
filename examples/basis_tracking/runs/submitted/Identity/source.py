import torch

class Identity:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.B = torch.eye(n, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        return self.B
