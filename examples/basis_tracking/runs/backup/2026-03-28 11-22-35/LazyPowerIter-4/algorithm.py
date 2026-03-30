import torch


class BasisUpdater:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device="cuda")
        self.B = torch.eye(m, device="cuda")  # orthogonal basis
        self.initialized = False
        self.update_every = 4  # update B every k steps
        self.step_count = 0

    def step(self, U: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        # Update accumulator A
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)

        if not self.initialized:
            # Initialize B via eigendecomposition on first step
            _, self.B = torch.linalg.eigh(self.A)
            self.initialized = True
            return self.B

        # Update B only every update_every steps
        if self.step_count % self.update_every == 0:
            # One step of orthogonal iteration: B <- Q from QR(A @ B)
            Z = self.A @ self.B
            Q, R = torch.linalg.qr(Z)
            self.B = Q

        return self.B
