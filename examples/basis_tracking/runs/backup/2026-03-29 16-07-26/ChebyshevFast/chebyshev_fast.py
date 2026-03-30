import torch


class BasisUpdater:
    """
    Fast Chebyshev-accelerated subspace iteration.
    
    Optimizations:
    1. Use inplace operations where possible
    2. Minimize temporaries
    3. Freq=3
    """
    
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.m = m
        self.A = torch.zeros(m, m, device='cuda')
        # Initialize B as random orthogonal
        B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(B)
        self.step_count = 0
        # Preallocate buffers
        self.AB = torch.empty_like(self.B)
        self.A2B = torch.empty_like(self.B)
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        """Update A and maintain approximate diagonalization."""
        # Update A
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        self.step_count += 1
        
        # Update frequency 3
        if self.step_count % 3 == 1 or self.step_count <= 2:
            # Chebyshev step: 2*A^2 @ B - B
            # Reuse preallocated buffers
            torch.matmul(self.A, self.B, out=self.AB)
            torch.matmul(self.A, self.AB, out=self.A2B)
            
            # Chebyshev combination inplace
            self.A2B.mul_(2.0).sub_(self.B)
            
            # Orthogonalize
            self.B, _ = torch.linalg.qr(self.A2B)
        
        return self.B
