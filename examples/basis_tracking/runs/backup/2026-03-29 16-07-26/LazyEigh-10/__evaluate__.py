import copy
import math
import random
import time
from collections import defaultdict

# pylint:disable=not-callable
import numpy as np
import torch
from click import echo  # use instead of print
from myautoresearch.evaluator import Evaluator


class BasisUpdater:
    def __init__(self, m: int, beta: float = 0.95):
        self.beta = beta
        self.A = torch.zeros(m, m, device='cuda')

    def step(self, U: torch.Tensor) -> torch.Tensor:
        """Updates accumulator A using linear interpolation: `A <- A * beta + (U U^T) * (1 - beta)`;
        Updates B (basis of A). Returns B.
        """
        UUt = U @ U.T
        self.A.lerp_(UUt, 1-self.beta)
        ...


def random_matrix(m, n, generator, rng: random.Random):
    matrix = torch.randn(m, n, device='cuda', generator=generator)
    matrix *= rng.triangular(-10, 10, 0) ** 2
    matrix += rng.triangular(-10, 10, 0) ** 2

    row = rng.randrange(0, m)
    matrix[row] *= rng.triangular(-10, 10, 0) ** 2
    matrix[row] += rng.triangular(-10, 10, 0) ** 2

    col = rng.randrange(0, n)
    matrix[:, col] *= rng.triangular(-10, 10, 0) ** 2
    matrix[:, col] += rng.triangular(-10, 10, 0) ** 2

    idx = (rng.randrange(0, m), rng.randrange(0, n))
    matrix[idx[0], idx[1]] *= rng.triangular(-10, 10, 0) ** 2
    matrix[idx[0], idx[1]] += rng.triangular(-10, 10, 0) ** 2

    return matrix


class BasisEvaluator(Evaluator):
    object: type[BasisUpdater]

    @torch.inference_mode()
    def run(self):
        generator = torch.Generator('cuda').manual_seed(0)
        rng = random.Random(0)

        shapes = [
            (2000, 2000),
            (2000, 1000),
            (2000, 500),
            (2000, 200),
            (2000, 100),
            (2000, 10),
            (2000, 1),
            (10, 2000),
            (100, 2000),
        ]

        mean_metrics = defaultdict(list)

        for m, n in shapes:
            I = torch.eye(m, device='cuda')

            # warmup
            warmup_updater = copy.deepcopy(self.object)(m, 0.95)
            for i in range(10):
                U = random_matrix(m,n, generator, rng)
                B = warmup_updater.step(U)

            # benchmark
            del warmup_updater
            updater = copy.deepcopy(self.object)(m, 0.95)
            A_ref = torch.zeros(m, m, device='cuda')

            for i in range(200):
                U = random_matrix(m,n, generator, rng)

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                B = updater.step(U)
                ender.record() # type:ignore
                torch.cuda.synchronize()
                sec = 1e-3 * starter.elapsed_time(ender)
                self.log_step(i, f"time per iter ({m}x{n})", sec)

                A_ref.lerp_(U @ U.T, 1-updater.beta)

                # check orthogonality
                ortho = float(torch.linalg.norm(B @ B.T - I).item())
                if not math.isfinite(ortho):
                    raise RuntimeError(f"Algorithm returned a matrix with nan or infinite values for matrix {m}x{n}.")

                self.log_step(i, f"orthogonality error ({m}x{n})", ortho)

                # check commutativity
                BtAB = B.T @ A_ref @ B
                diag_norm = torch.linalg.vector_norm(BtAB.diagonal())
                full_norm = torch.linalg.vector_norm(BtAB).clip(min=1e-10)
                expl_var = float((diag_norm / full_norm).item())

                self.log_step(i, f"explained variance ({m}x{n})", expl_var) # score is 0 to 1.
                self.file_logger.info("%ix%i %i: sec=%.3f, ortho=%.8f, expl.var=%.8f", m, n, i, sec, ortho, expl_var)

            del updater, A_ref, I

        for k, maximize, weight, display_rank in [
            ("time per iter", False, 1, False),
            ("orthogonality error", False, 1, False),
            ("explained variance", True, 4, True),
        ]:
            for m,n in shapes:
                value = self.history.mean(f"{k} ({m}x{n})")
                mean_metrics[k].append(value)
                self.log_final(f"{k} ({m}x{n})", value, maximize=maximize, display_leaderboard=False, display_summary=False, weight=weight, is_main=False, display_rank=display_rank)

            # log mean metrics
            self.log_final(f"mean {k}", np.mean(mean_metrics[k]), maximize=maximize, is_main=False, display_rank=display_rank)

        t = self._metrics["mean time per iter"].value
        o = self._metrics["mean orthogonality error"].value
        v = self._metrics["mean explained variance"].value
        score = (v ** 10) / ((o + 1e-3) * (t + 1e-2) * 1000)
        self.log_final("score", score, maximize=True)

if __name__ == "__main__":
    evaluator = BasisEvaluator()
    evaluator.run()
    evaluator._save()
