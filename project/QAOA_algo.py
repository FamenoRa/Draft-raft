# For quantum optimization setup

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from qiskit_aer import Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import Sampler

def optimize_light_cycle(veh_counts: dict, cycle_time: int, min_green: int = 5) -> dict:
    """
    Optimize integer green times for 4 junctions using QAOA.
    veh_counts: dictionary mapping junction_id to vehicle count
    cycle_time: total cycle time in seconds
    min_green: minimum green time per junction
    Returns dict with green times per junction and status.
    """
    print(veh_counts)
    # Build cost coefficients inversely proportional to traffic
    costs = {j: 1/(cnt + 1e-6) for j, cnt in veh_counts.items()}

    qp = QuadraticProgram(name='Intersection_cycle')
    # Define integer vars x1..x4 for each junction green time
    max_individual = cycle_time - min_green * (len(veh_counts) - 1)
    for j in veh_counts.keys():
        qp.integer_var(name=f'x{j}', lowerbound=min_green, upperbound=max_individual)

    # Constraint: sum of all green times == cycle_time
    linear_coeffs = {f'x{j}': 1 for j in veh_counts.keys()}
    qp.linear_constraint(linear_coeffs, '==', cycle_time, name='cycle_sum')

    # Objective: minimize sum_i cost_i * x_i
    qp.minimize(linear={f'x{j}': cost for j, cost in costs.items()})

    # Solve with QAOA optimizer
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=20), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    # Extract green times
    greens = {j: int(result.variables_dict[f'x{j}']) for j in veh_counts.keys()}
    greens['status'] = result.status.name
    return greens