import torch
import torch.nn.functional as F


class BasisUpdater:
    """Subspace iteration with frequency 3 and 2 power iterations."""
    
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.m = m
        self.A = torch.zeros(m, m, device='cuda')
        # Random orthogonal initialization
        B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(B)
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        """Updates A and returns B which approximately diagonalizes A."""
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        self.step_count += 1
        
        # Update every 3 steps with 2 power iterations
        if self.step_count % 3 == 1 or self.step_count <= 2:
            # Two rounds of power iteration for better convergence
            AB = self.A @ self.B
            AB = self.A @ AB
            self.B, _ = torch.linalg.qr(AB)
        
        return self.B
