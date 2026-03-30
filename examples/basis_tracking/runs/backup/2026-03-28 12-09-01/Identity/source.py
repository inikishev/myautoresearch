import torch

class Identity:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.B = torch.eye(n, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        return self.B


class RandomOrtho:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.B = torch.linalg.qr(torch.randn(n, n, device='cuda')).Q

    def step(self, U: torch.Tensor) -> torch.Tensor:
        return self.B

class Eigh:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(n, n, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        L, Q = torch.linalg.eigh(self.A)
        return Q

class LazyEigh:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(n, n, device='cuda')
        self.B = None
        self.current_step = 0

    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)

        if self.current_step % 10 == 0:
            L, self.B = torch.linalg.eigh(self.A)

        self.current_step += 1
        return self.B


class PowerIter:
    def __init__(self, n: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(n, n, device='cuda')
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
