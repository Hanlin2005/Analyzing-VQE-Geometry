"""
In this code, I analyze the impact of various hardware geometries and SWAP
induced noise on VQE of 1D and 2D Heisenberg Models across various optimization schemes
"""

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


algorithm_globals.random_seed = 420
np.random.seed(420)

def create_heisenberg_hamiltonian(geometry, num_qubits, J):
    """
    Creates a Heisenberg Hamiltonian H = J[0]*XX + J[1]*YY + J[2]*ZZ
    geometry: '1D' (line) or '2D' (square/grid)
    """
    Jx, Jy, Jz = J
    
    if geometry == '1D':
        pairs = [(i, i+1) for i in range(num_qubits - 1)]
    elif geometry == '2D':
        pairs = [(0, 1), (1, 3), (3, 2), (2, 0)]

    op_list = []
    for (i, j) in pairs:
        if Jx != 0: op_list.append(("XX", [i, j], Jx))
        if Jy != 0: op_list.append(("YY", [i, j], Jy))
        if Jz != 0: op_list.append(("ZZ", [i, j], Jz))
            
    return SparsePauliOp.from_sparse_list(op_list, num_qubits=num_qubits)

def get_exact_energy(hamiltonian):
    """Calculates the classical exact minimum eigenvalue for comparison."""
    matrix = hamiltonian.to_matrix()
    vals = np.linalg.eigvalsh(matrix)
    return vals[0]

def get_hardware_coupling_map(geometry, num_qubits):
    if geometry == '1D':
        return [[i, i+1] for i in range(num_qubits-1)] + [[i+1, i] for i in range(num_qubits-1)]
    elif geometry == '2D':
        connections = [[0, 1], [1, 3], [3, 2], [2, 0]]
        return connections + [[v, u] for u, v in connections]

def create_noise_model(error_rate=0.05):
    """
    Creates a noise model that specifically targets the CNOT gates.
    """
    noise_model = NoiseModel()
    error_gate = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_gate, ["cx"])
    return noise_model


def run_vqe_trial(hamiltonian, coupling_map, optimizer_name, problem_geometry, noise_model=None):
    """
    Runs a single VQE trial. 
    If noise_model is None, runs an ideal simulation (still respecting coupling map).
    """
    # 1. Define Ansatz based on Problem Geometry (FIXED LOGIC)
    if problem_geometry == '1D':
        ent_strategy = [(0,1), (1,2), (2,3)] 
    elif problem_geometry == '2D':
        ent_strategy = [(0,1), (1,3), (3,2), (2,0)]
        
    ansatz = EfficientSU2(num_qubits=4, su2_gates=['ry'], entanglement=ent_strategy, reps=2)
    
    # 2. Define Optimizer
    if optimizer_name == 'COBYLA':
        optimizer = COBYLA(maxiter=80)
    elif optimizer_name == 'SPSA':
        optimizer = SPSA(maxiter=80)
    elif optimizer_name == 'L_BFGS_B':
        optimizer = L_BFGS_B(maxiter=50)
    else:
        raise ValueError("Unknown Optimizer")

    # 3. Define Backend
    backend = AerSimulator(noise_model=noise_model, coupling_map=coupling_map)
    backend.set_options(shots=1024) 
    
    # 4. Transpile for COUNTING (Allow 'swap' so we can count them)
    qc_for_counting = transpile(
        ansatz, 
        coupling_map=coupling_map, 
        optimization_level=1, 
        basis_gates=['cx', 'id', 'rz', 'sx', 'x', 'ry', 'swap'], # Allow swaps
        initial_layout=list(range(4))
    )
    swap_count = qc_for_counting.count_ops().get('swap', 0)

    # 5. Transpile for RUNNING (Decompose 'swap' into CNOTs so noise hits them)
    qc_for_running = transpile(
        ansatz, 
        coupling_map=coupling_map, 
        optimization_level=1, 
        basis_gates=['cx', 'id', 'rz', 'sx', 'x', 'ry'], # NO swaps allowed
        initial_layout=list(range(4))
    )
    
    # 6. Setup VQE
    counts = []
    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
    
    noisy_estimator = BackendEstimatorV2(backend=backend)
    
    vqe = VQE(noisy_estimator, ansatz, optimizer, callback=store_intermediate_result)
    
    # IMPORTANT: Run the circuit that has CNOTs (qc_for_running)
    vqe.ansatz = qc_for_running
    
    start_time = time.time()
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    end_time = time.time()
    
    return {
        "eigenvalue": result.eigenvalue.real,
        "convergence_counts": counts,
        "convergence_values": values,
        "duration": end_time - start_time,
        "depth": qc_for_running.depth(),
        "swaps": swap_count
    }

if __name__ == "__main__":
    print("Initializing Quantum Engineering Final Project...")
    
    num_qubits = 4
    coeffs = (1.0, 0.5, 0.5)
    
    problem_geometries = ['1D', '2D']
    hardware_geometries = ['1D', '2D']
    optimizers = ['COBYLA', 'SPSA'] 
    
    results_db = {}
    
    noise_model = create_noise_model(error_rate=0.05)

    print(f"\nModel: Heisenberg aXX+bYY+cZZ with a={coeffs[0]}, b={coeffs[1]}, c={coeffs[2]}")
    print("-" * 60)

    # --- EXPERIMENT LOOP ---
    for prob_geo in problem_geometries:
        H = create_heisenberg_hamiltonian(prob_geo, num_qubits, coeffs)
        exact_energy = get_exact_energy(H)
        print(f"\n>>> Problem Geometry: {prob_geo} | Exact Ground State: {exact_energy:.4f} Ha")
        
        for hw_geo in hardware_geometries:
            cmap = get_hardware_coupling_map(hw_geo, num_qubits)
            
            for opt in optimizers:
                # Key for identifying this configuration
                config_key = f"P:{prob_geo}_H:{hw_geo}_{opt}"
                print(f"    Running {config_key}...", end="", flush=True)
                
                # 1. Run Noisy
                res_noisy = run_vqe_trial(H, cmap, opt, prob_geo, noise_model)
                
                # 2. Run Ideal (No Noise Model)
                res_ideal = run_vqe_trial(H, cmap, opt, prob_geo, noise_model=None)
                
                print(f" Done (Swaps: {res_noisy['swaps']})")
                
                results_db[config_key] = {
                    'noisy': res_noisy,
                    'ideal': res_ideal,
                    'exact': exact_energy,
                    'prob_geo': prob_geo,
                    'hw_geo': hw_geo,
                    'optimizer': opt
                }


#ANALYSIS
    # Define labels helper
    keys = list(results_db.keys())
    labels = []
    for k in keys:
        parts = k.split('_')
        prob_g = parts[0].split(':')[1]
        hw_g = parts[1].split(':')[1]
        opt_name = parts[2]
        # Multi-line label
        label = f"Problem: {prob_g}\nHardware: {hw_g}\n{opt_name}"
        labels.append(label)
    
    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, opt in enumerate(optimizers):
        ax = axes[idx]
        ax.set_title(f"Convergence (Noisy): {opt}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Energy (Ha)")
        
        for key, data in results_db.items():
            if data['optimizer'] == opt:
                label = f"P:{data['prob_geo']} H:{data['hw_geo']}"
                ax.plot(data['noisy']['convergence_counts'], data['noisy']['convergence_values'], label=label)
                ax.axhline(y=data['exact'], color='k', linestyle='--', alpha=0.1)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vqe_convergence_comparison.png')
    print("Saved: vqe_convergence_comparison.png")
    plt.close()

    # --- B. Noise Impact Analysis (Error Bar Chart) ---
    plt.figure(figsize=(14, 7))
    
    ideal_errors = [abs(results_db[k]['ideal']['eigenvalue'] - results_db[k]['exact']) for k in keys]
    noisy_errors = [abs(results_db[k]['noisy']['eigenvalue'] - results_db[k]['exact']) for k in keys]
    
    plt.bar(x - width/2, ideal_errors, width, label='Ideal (Topology Only)', color='skyblue', alpha=0.9, edgecolor='black')
    plt.bar(x + width/2, noisy_errors, width, label='Noisy (Topology + Gate Error)', color='salmon', alpha=0.9, edgecolor='black')
    
    plt.ylabel('Absolute Energy Error (Hartree)', fontsize=12)
    plt.title('Impact of Hardware Noise vs. Topology Constraints', fontsize=14)
    plt.xticks(x, labels, rotation=0, fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('noise_impact_analysis.png')
    print("Saved: noise_impact_analysis.png")
    plt.close()
    
    #SWAP Count
    plt.figure(figsize=(14, 7))
    
    swap_counts = [results_db[k]['noisy']['swaps'] for k in keys]
    
    # Simple bar chart for swaps
    bars = plt.bar(x, swap_counts, width=0.6, color='thistle', edgecolor='black', alpha=0.9)
    
    plt.ylabel('Swap Gate Count', fontsize=12)
    plt.title('Compilation Overhead: SWAP Gates Inserted', fontsize=14)
    plt.xticks(x, labels, rotation=0, fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig('topology_overhead_analysis.png')
    print("Saved: topology_overhead_analysis.png")
    plt.close()

    header = ["Trial Configuration", "Exact Energy", "Ideal Energy", "Noisy Energy", "Ideal Error", "Noisy Error", "Depth", "Swaps"]
    
    print("\n" + "="*115)
    print(f"{header[0]:<25} | {header[1]:<10} | {header[2]:<10} | {header[3]:<10} | {header[4]:<10} | {header[5]:<10} | {header[6]:<6} | {header[7]:<6}")
    print("-" * 115)
    
    csv_rows = []
    
    for key, data in results_db.items():
        exact = data['exact']
        e_ideal = data['ideal']['eigenvalue']
        e_noisy = data['noisy']['eigenvalue']
        err_ideal = abs(e_ideal - exact)
        err_noisy = abs(e_noisy - exact)
        
        depth = data['noisy']['depth']
        swaps = data['noisy']['swaps']
        
        row_str = f"{key:<25} | {exact:.4f}     | {e_ideal:.4f}     | {e_noisy:.4f}     | {err_ideal:.4f}     | {err_noisy:.4f}     | {depth:<6} | {swaps:<6}"
        print(row_str)
        
        csv_rows.append([key, exact, e_ideal, e_noisy, err_ideal, err_noisy, depth, swaps])
        
    print("="*115 + "\n")

    with open('accuracy_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print("Saved: accuracy_table.csv")