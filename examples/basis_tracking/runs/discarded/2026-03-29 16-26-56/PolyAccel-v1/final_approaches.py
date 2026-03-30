"""
Final set of unusual approaches: tensor methods, implicit restarts, and optimized variants.
"""
import torch


class TensorSketchDiagonalizer:
    """
    Use tensor sketching for approximate matrix operations.
    Very unusual approach from randomized linear algebra.
    """
    def __init__(self, m: int, beta: float = 0.95, sketch_size: int = 300):
        self.m = m
        self.beta = beta
        self.sketch_size = min(sketch_size, m)
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        # Random sketching matrices
        self.S1 = torch.randn(m, self.sketch_size, device='cuda') / (self.sketch_size ** 0.5)
        self.S2 = torch.randn(m, self.sketch_size, device='cuda') / (self.sketch_size ** 0.5)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Sketch A: AS1 and AS2
            AS1 = self.A @ self.S1  # m x sketch_size
            
            # Compute eigenvectors of sketched matrix
            # Small eigendecomposition: (AS1)^T (AS1)
            small = AS1.T @ AS1  # sketch_size x sketch_size
            eigvals, V = torch.linalg.eigh(small)
            
            # Lift to full space: AS1 @ V
            top_k = AS1 @ V  # m x sketch_size
            
            # Power iteration refinement
            for _ in range(1):
                top_k = self.A @ top_k
                top_k, _ = torch.linalg.qr(top_k)
            
            # Pad to full rank
            if top_k.shape[1] < self.m:
                pad = torch.randn(self.m, self.m - top_k.shape[1], device='cuda')
                pad = pad - top_k @ (top_k.T @ pad)
                pad, _ = torch.linalg.qr(pad)
                self.B = torch.cat([top_k, pad], dim=1)
                self.B, _ = torch.linalg.qr(self.B)
            else:
                self.B, _ = torch.linalg.qr(top_k)
        
        return self.B


class ImplicitRestart:
    """
    Implicitly restarted Arnoldi/Lanczos method.
    Maintains a fixed-size Krylov subspace and restarts implicitly.
    """
    def __init__(self, m: int, beta: float = 0.95, krylov_size: int = 100):
        self.m = m
        self.beta = beta
        self.krylov_size = min(krylov_size, m)
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        # Krylov basis
        self.V = torch.randn(m, self.krylov_size, device='cuda')
        self.V, _ = torch.linalg.qr(self.V)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Implicit restart: extend Krylov subspace
            v_new = self.A @ self.V[:, -1:]
            v_new = v_new - self.V @ (self.V.T @ v_new)
            v_new = v_new / (torch.norm(v_new) + 1e-8)
            
            # Update basis (shift oldest out)
            self.V = torch.cat([self.V[:, 1:], v_new], dim=1)
            
            # Rayleigh-Ritz
            H = self.V.T @ self.A @ self.V
            eigvals, Y = torch.linalg.eigh(H)
            
            # Ritz vectors
            top_k = self.V @ Y[:, -self.krylov_size:]
            
            # Pad to full rank
            pad = torch.randn(self.m, self.m - self.krylov_size, device='cuda')
            pad = pad - top_k @ (top_k.T @ pad)
            pad, _ = torch.linalg.qr(pad)
            self.B = torch.cat([top_k, pad], dim=1)
            self.B, _ = torch.linalg.qr(self.B)
        
        return self.B


class FastBlockKrylov:
    """
    Optimized block Krylov method with smaller blocks and fewer iterations.
    """
    def __init__(self, m: int, beta: float = 0.95, block_size: int = 50, krylov_dim: int = 2):
        self.m = m
        self.beta = beta
        self.block_size = min(block_size, m)
        self.krylov_dim = krylov_dim
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Block Krylov with warm start from current B
            Q = self.B[:, :self.block_size].clone()
            K_blocks = [Q]
            
            for _ in range(self.krylov_dim - 1):
                Q = self.A @ Q
                Q, _ = torch.linalg.qr(Q)
                K_blocks.append(Q)
            
            K = torch.cat(K_blocks, dim=1)
            K, _ = torch.linalg.qr(K)
            
            # Rayleigh-Ritz
            AK = K.T @ self.A @ K
            eigvals, V = torch.linalg.eigh(AK)
            
            top_k = K @ V
            
            # Pad to full rank
            if top_k.shape[1] < self.m:
                pad = torch.randn(self.m, self.m - top_k.shape[1], device='cuda')
                pad = pad - top_k @ (top_k.T @ pad)
                pad, _ = torch.linalg.qr(pad)
                self.B = torch.cat([top_k, pad], dim=1)
                self.B, _ = torch.linalg.qr(self.B)
            else:
                self.B, _ = torch.linalg.qr(top_k)
        
        return self.B


class HybridPowerKrylov:
    """
    Hybrid: power iteration for most updates, Krylov occasionally.
    """
    def __init__(self, m: int, beta: float = 0.95, krylov_freq: int = 10):
        self.m = m
        self.beta = beta
        self.krylov_freq = krylov_freq
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % self.krylov_freq == 0:
            # Full Krylov update
            block_size = 100
            Q = torch.randn(self.m, block_size, device='cuda')
            Q, _ = torch.linalg.qr(Q)
            
            K = [Q]
            for _ in range(2):
                Q = self.A @ Q
                Q, _ = torch.linalg.qr(Q)
                K.append(Q)
            
            K_mat = torch.cat(K, dim=1)
            K_mat, _ = torch.linalg.qr(K_mat)
            
            AK = K_mat.T @ self.A @ K_mat
            eigvals, V = torch.linalg.eigh(AK)
            
            top_k = K_mat @ V[:, -min(self.m, K_mat.shape[1]):]
            
            if top_k.shape[1] < self.m:
                pad = torch.randn(self.m, self.m - top_k.shape[1], device='cuda')
                pad = pad - top_k @ (top_k.T @ pad)
                pad, _ = torch.linalg.qr(pad)
                self.B = torch.cat([top_k, pad], dim=1)
                self.B, _ = torch.linalg.qr(self.B)
            else:
                self.B, _ = torch.linalg.qr(top_k)
        
        elif self.step_count % 3 == 0:
            # Regular power iteration
            for _ in range(2):
                self.B = torch.linalg.qr(self.A @ self.B)[0]
        
        return self.B


class StreamingRayleighRitz:
    """
    Streaming Rayleigh-Ritz: maintain a small subspace and update it incrementally.
    """
    def __init__(self, m: int, beta: float = 0.95, subspace_size: int = 150):
        self.m = m
        self.beta = beta
        self.subspace_size = min(subspace_size, m)
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        # Subspace
        self.Q = torch.randn(m, self.subspace_size, device='cuda')
        self.Q, _ = torch.linalg.qr(self.Q)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Extend subspace with A @ Q
            AQ = self.A @ self.Q
            
            # Orthogonalize against existing Q
            AQ = AQ - self.Q @ (self.Q.T @ AQ)
            AQ, _ = torch.linalg.qr(AQ)
            
            # Combine old Q with new directions
            combined = torch.cat([self.Q, AQ[:, :self.subspace_size//2]], dim=1)
            self.Q, _ = torch.linalg.qr(combined)
            
            # Keep only subspace_size columns
            if self.Q.shape[1] > self.subspace_size:
                self.Q = self.Q[:, :self.subspace_size]
            
            # Rayleigh-Ritz
            H = self.Q.T @ self.A @ self.Q
            eigvals, V = torch.linalg.eigh(H)
            
            top_k = self.Q @ V
            
            # Pad to full rank
            pad = torch.randn(self.m, self.m - top_k.shape[1], device='cuda')
            pad = pad - top_k @ (top_k.T @ pad)
            pad, _ = torch.linalg.qr(pad)
            self.B = torch.cat([top_k, pad], dim=1)
            self.B, _ = torch.linalg.qr(self.B)
        
        return self.B


class OptimizedSubspaceIter:
    """
    Optimized subspace iteration with single QR for multiple iterations.
    Similar to PowerIterFreq3 but with different frequency/iteration trade-off.
    """
    def __init__(self, m: int, beta: float = 0.95, freq: int = 4, n_iter: int = 2):
        self.m = m
        self.beta = beta
        self.freq = freq
        self.n_iter = n_iter
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % self.freq == 0:
            # Multiple power iterations, single QR at the end
            Y = self.B.clone()
            for _ in range(self.n_iter):
                Y = self.A @ Y
            self.B = torch.linalg.qr(Y)[0]
        
        return self.B


class RandomizedCoordinateDescent:
    """
    Randomized coordinate descent on Stiefel manifold.
    Update one coordinate (column of B) at a time.
    """
    def __init__(self, m: int, beta: float = 0.95, n_updates: int = 50):
        self.m = m
        self.beta = beta
        self.n_updates = n_updates
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        self.step_count = 0
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Randomized coordinate updates
            for _ in range(self.n_updates):
                # Pick random column
                j = torch.randint(0, self.m, (1,)).item()
                
                # Update column j: move toward A @ B[:, j]
                new_col = self.A @ self.B[:, j]
                
                # Orthogonalize against other columns
                mask = torch.ones(self.m, dtype=torch.bool, device='cuda')
                mask[j] = False
                new_col = new_col - self.B[:, mask] @ (self.B[:, mask].T @ new_col)
                new_col = new_col / (torch.norm(new_col) + 1e-8)
                
                self.B[:, j] = new_col
        
        return self.B


class PolynomialAccelerated:
    """
    Polynomial-accelerated power iteration.
    Use Chebyshev-like polynomials for faster convergence.
    """
    def __init__(self, m: int, beta: float = 0.95, degree: int = 2):
        self.m = m
        self.beta = beta
        self.degree = degree
        self.A = torch.zeros(m, m, device='cuda')
        
        # Random orthogonal init
        self.B = torch.randn(m, m, device='cuda')
        self.B, _ = torch.linalg.qr(self.B)
        
        # Eigenvalue estimates for Chebyshev scaling
        self.lambda_max = 1.0
        self.lambda_min = 0.0
        
        self.step_count = 0
        
    def _chebyshev_step(self, X, c, d):
        """Apply Chebyshev polynomial: T_2(A) @ X = 2*A^2 @ X - X (approximately)"""
        # Simplified: just use power iteration with coefficient
        return c * self.A @ X + d * X
        
    def step(self, U: torch.Tensor) -> torch.Tensor:
        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        
        # Update eigenvalue estimates
        if self.step_count % 10 == 0:
            v = torch.randn(self.m, device='cuda')
            v = v / torch.norm(v)
            for _ in range(5):
                v = self.A @ v
                v = v / torch.norm(v)
            self.lambda_max = (v @ (self.A @ v)).item()
        
        self.step_count += 1
        
        if self.step_count % 3 == 0:
            # Polynomial-accelerated iteration
            # T_2(A) = 2*A^2 - I (scaled)
            # Apply T_2(A) @ B
            AB = self.A @ self.B
            
            # Estimate scale
            scale = 2.0 / (self.lambda_max + 0.1)
            
            Y = scale * self.A @ AB - self.B
            self.B = torch.linalg.qr(Y)[0]
        
        return self.B
