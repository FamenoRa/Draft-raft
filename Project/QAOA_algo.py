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

def optimize_light_cycle(total_vehicles: int, main_ratio: float, cycle_time: int,
                         min_green: int = 10) -> dict:
    """
    Optimize integer green times for main and side roads using QAOA.
    Returns dict with 'main_green', 'side_green', and 'status'.
    """
    # Compute weights inversely proportional to traffic
    main_veh = total_vehicles * main_ratio
    side_veh = total_vehicles * (1-main_ratio)
    cost_main = 1/(main_veh+1e-6)
    cost_side = 1/(side_veh+1e-6)

    # Build QUBO with integer variables x0, x1
    qp = QuadraticProgram(name='traffic_cycle')
    qp.integer_var(name='x0', lowerbound=min_green, upperbound=cycle_time-min_green)  # x0 = main road green time
    qp.integer_var(name='x1', lowerbound=min_green, upperbound=cycle_time-min_green)  # x1 = side road green time
    # Constraint: x0 + x1 == cycle_time
    qp.linear_constraint({'x0':1,'x1':1}, '==', cycle_time, name='cycle_sum')
    # Objective: minimize cost_main*x0 + cost_side*x1
    qp.minimize(linear={'x0': cost_main, 'x1': cost_side})

    # Solve with QAOA
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=25), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    # Extract solution
    mg = int(result.variables_dict['x0'])
    sg = int(result.variables_dict['x1'])
    return {'main_green': mg, 'side_green': sg, 'status': result.status.name}