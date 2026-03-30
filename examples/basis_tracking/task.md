## Task

We have an accumulator A; on each step it is updated with correction $U U^T$ via the exponential moving average formula: $A \leftarrow A * \beta + (U U^T) * (1 - \beta)$. The goal is to maintain an orthogonal matrix B which approximately diagonalizes A. The issue with eigendecomposition and power iteration through QR is that both are expensive for large matrices and do not take advantage of the structure of the problem. Your goal is to develop a faster method for maintaining an approximate diagonalization.

## API

Submit a class that has the following interface:

```py
class BasisUpdater:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        """Updates accumulator A using linear interpolation: `A <- A * beta + (U U^T) * (1 - beta)`;
        Returns B which approximately diagonalizes updated A.
        """
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)
        ...
        # returns eigenbasis which has same shape as UUt
```

Pass name of the class to the evaluation script, for example `--object BasisUpdater`.

## Evaluation

Evaluation runs the algorithm for 200 steps on U of various sizes ranging from 2000x2000 to 2000x1 and 10x2000. It computes orthogonality error as average $||B B^T - I||_F$ over all steps, and explained variance as average $\frac{||diag(B^T A B)||_F}{||B^T A B||_F}$ (fraction of total Frobenius norm pushed into the diagonal). The evaluation script keeps track of its own copy of A, so make sure you update A correctly. The evaluation script has a hard timeout of 360 seconds (for reference the eigendecomposition every step baseline takes 290 seconds).

The submissions are scored using the following formula (higher is better):

`score = (explained_variance ^ 10) / ((orthogonality_error + 1e-2) * (time_per_iter + 1e-2) * 1000)`
