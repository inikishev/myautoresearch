import torch


class LazySubspaceIter:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')
        self.B = None
        self.current_step = 0
        self.update_freq = 3

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        if self.B is None:
            L, self.B = torch.linalg.eigh(self.A)

        elif self.current_step % self.update_freq == 0:
            subspace_iter = self.A @ self.B
            self.B, R = torch.linalg.qr(subspace_iter)

        self.current_step += 1
        return self.B
