import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import BackendEstimatorV2
import time
import csv

print("\n--- VISUAL PROOF OF SWAPS ---")
# Create a mismatched case: 2D Ansatz (Ring) on 1D Hardware (Line)
ansatz_2d = EfficientSU2(4, entanglement=[(0,1), (1,3), (3,2), (2,0)], reps=1)
hw_map_1d = [[0,1], [1,0], [1,2], [2,1], [2,3], [3,2]]

qc = transpile(ansatz_2d, coupling_map=hw_map_1d, 
               basis_gates=['cx', 'ry', 'swap'], # <--- The secret sauce
               initial_layout=[0,1,2,3],
               optimization_level=1)

print(f"Swaps found: {qc.count_ops().get('swap', 0)}")
print(qc.draw(output='text', idle_wires=False))