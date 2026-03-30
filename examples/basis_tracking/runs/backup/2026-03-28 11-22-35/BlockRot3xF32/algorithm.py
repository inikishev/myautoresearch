import torch


class BasisUpdater:
    """Novel approach: Block rotation (3x) with contiguous memory.

    Uses contiguous() to ensure memory access patterns are optimized.
    Block size = k * 3, PI every 6 for small_k, 7 for large_m.
    """

    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device="cuda", dtype=torch.float32)
        self.m = m
        self.B: torch.Tensor | None = None
        self.step_count = 0

    def step(self, U: torch.Tensor) -> torch.Tensor:
        k = U.shape[1]
        m = self.m

        UUt = U @ U.T
        self.A.lerp_(UUt, 1 - self.beta)
        self.step_count += 1

        if self.B is None:
            _, self.B = torch.linalg.eigh(self.A)
            return self.B

        if m <= 500:
            _, self.B = torch.linalg.eigh(self.A)
            return self.B

        if k < m:
            return self._block_rotation_small_k(k)
        else:
            return self._lazy_pi_large_m()

    def _block_rotation_small_k(self, k: int) -> torch.Tensor:
        assert self.B is not None

        block_size = min(k * 3, self.m)
        B_block = self.B[:, :block_size].contiguous()
        C = B_block.T @ self.A @ B_block
        _, evecs = torch.linalg.eigh(C)
        self.B[:, :block_size] = B_block @ evecs

        if self.step_count % 6 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q

        return self.B

    def _lazy_pi_large_m(self) -> torch.Tensor:
        assert self.B is not None

        if self.step_count % 7 == 0:
            Z = self.A @ self.B
            Q, _ = torch.linalg.qr(Z)
            self.B = Q
        return self.B