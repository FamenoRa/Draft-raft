# For quantum optimization setup

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_primitives import Sampler

def setup_qaoa_optimizer(reps: int = 2, maxiter: int = 100) -> MinimumEigenOptimizer:
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=SPSA(maxiter=maxiter), reps=reps)
    return MinimumEigenOptimizer(qaoa)