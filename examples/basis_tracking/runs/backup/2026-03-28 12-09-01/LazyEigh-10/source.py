import torch

class LazyEigh:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')
        self.B = None
        self.current_step = 0

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        if self.current_step % 10 == 0:
            L, self.B = torch.linalg.eigh(self.A)

        self.current_step += 1
        return self.B

