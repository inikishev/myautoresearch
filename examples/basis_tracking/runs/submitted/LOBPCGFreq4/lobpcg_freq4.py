import torch


class BasisUpdater:
    """LOBPCG + power iteration with freq=4."""
    
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.m = m
        self.A = torch.zeros(m, m, device='cuda')
        self.B = None
        self.step_count = 0
        self.freq = 4  # Higher frequency = less computation
        self.use_lobpcg = m <= 100
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        """Updates accumulator A and returns B which approximately diagonalizes A."""
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        self.step_count += 1
        
        # Initialize B on first step
        if self.B is None:
            B = torch.randn(self.m, self.m, device='cuda', dtype=self.A.dtype)
            self.B, _ = torch.linalg.qr(B)
        
        # Only update every freq steps
        if self.step_count % self.freq == 1:
            if self.use_lobpcg and self.m >= 3:
                k = self.m // 3
                if k >= 1:
                    X = self.B[:, :k]
                    eigenvalues, eigenvectors = torch.lobpcg(
                        self.A,
                        k=k,
                        B=None,
                        X=X,
                        niter=2,
                        tol=1e-3,
                        largest=True,
                    )
                    self.B[:, :k] = eigenvectors
                    remainder = self.B[:, k:]
                    proj = eigenvectors @ (eigenvectors.T @ remainder)
                    remainder = remainder - proj
                    self.B[:, k:], _ = torch.linalg.qr(remainder)
            
            # Two power iterations with one QR
            AB = self.A @ self.B
            A2B = self.A @ AB
            self.B, _ = torch.linalg.qr(A2B)
        
        return self.B
