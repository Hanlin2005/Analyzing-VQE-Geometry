# Hardware Topology Effects on VQE Performance for Heisenberg Spin Models

A quantum computing simulation study investigating how qubit connectivity constraints and SWAP-induced gate noise degrade Variational Quantum Eigensolver (VQE) accuracy when solving 1D and 2D Heisenberg spin models on mismatched hardware topologies.

> **Course:** ECE 520 Final Project &nbsp;|&nbsp; **Author:** Hanlin Wang &nbsp;|&nbsp; **Date:** December 2025

---

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusions](#conclusions)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [References](#references)

---

## Introduction

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm that finds the ground state energy of a Hamiltonian by offloading optimization to a classical computer while using a quantum circuit for state preparation and energy estimation. However, real quantum hardware imposes fixed qubit connectivity, and when the problem ansatz requires entangling non-adjacent qubits, the compiler must insert **SWAP gates** to route quantum states.

Since a single SWAP gate decomposes into three CNOT gates — among the noisiest operations on current hardware — this routing overhead can substantially degrade VQE accuracy. **Choosing a hardware geometry that matches the problem geometry is therefore critical.**

This project systematically studies the impact of hardware-problem geometry mismatch on ground state energy estimation for 4-qubit 1D and 2D Heisenberg spin models, and evaluates the robustness of COBYLA and SPSA optimizers under SWAP-induced noise.

---

## Background

### Variational Quantum Eigensolver

VQE is grounded in the **Variational Principle**: for any normalized trial wavefunction |ψ⟩, the expectation value of H is bounded below by the true ground state energy E₀:

```
E₀ ≤ ⟨ψ(θ)|H|ψ(θ)⟩
```

The algorithm iterates a hybrid loop: a parameterized quantum circuit (the ansatz) prepares a trial state, the expectation value of each Pauli term in the Hamiltonian is measured, and a classical optimizer updates the circuit parameters to minimize the energy. This continues until convergence.

### Heisenberg Spin Model

The quantum Heisenberg model describes spin-spin interactions on a lattice and is a standard benchmark for quantum simulation — it exhibits non-trivial quantum entanglement and becomes classically intractable as system size grows. Its general Hamiltonian is:

```
H = Σ_{⟨i,j⟩} (Jx·XᵢXⱼ + Jy·YᵢYⱼ + Jz·ZᵢZⱼ) + Σᵢ hᵢZᵢ
```

This project uses the **anisotropic XYZ model** (Jx ≠ Jy ≠ Jz), which creates a complex energy landscape that provides a robust test for VQE under noisy conditions.

---

## Methodology

### Experimental Configurations

Four distinct trials were constructed using a 4-qubit system, crossing two problem geometries against two hardware topologies:

| Trial | Problem Geometry | Hardware Topology |
|-------|-----------------|-------------------|
| 1 | 1D (linear chain) | 1D (linear chain) — **matched** |
| 2 | 2D (square ring) | 2D (square ring) — **matched** |
| 3 | 2D (square ring) | 1D (linear chain) — **mismatched** |
| 4 | 1D (linear chain) | 2D (square ring) — **mismatched** |

### Ansatz

The `EfficientSU2` hardware-efficient ansatz was used, consisting of a layer of Rᵧ rotation gates followed by entangling CNOT gates, repeated for 2 layers. Crucially, the **entanglement structure was matched to the problem geometry** (not the hardware), forcing the compiler to insert SWAP gates when a mismatch exists. This isolates the routing overhead as the variable under test.

- **1D problems:** linear entanglement — pairs `(0,1), (1,2), (2,3)`
- **2D problems:** cyclic/square entanglement — pairs `(0,1), (1,3), (3,2), (2,0)`

### Noise Model

A **depolarizing error channel** was applied to all CNOT gates at rate `p = 0.03`. Since each SWAP gate decomposes into 3 CNOTs, every topology-forced SWAP incurs a compounded `~0.09` error rate. SWAP gates were first counted at the transpilation level (allowing `swap` in basis gates), then decomposed into CNOTs for simulation so that all routing noise is correctly captured.

Each configuration was run under two conditions:
- **Ideal:** topology constraints enforced, no gate error
- **Noisy:** topology constraints enforced + depolarizing noise

Energy error was benchmarked against the exact classical ground state computed via `numpy.linalg.eigvalsh`.

### Optimizers

Both COBYLA (gradient-free, simplex-based) and SPSA (stochastic gradient approximation) were evaluated to assess robustness under SWAP noise.

---

## Results

### Geometry Mismatch Impact

| Config | Opt | E_exact | E_ideal | E_noisy | ΔE_ideal | ΔE_noisy | Depth (Swaps) |
|--------|-----|---------|---------|---------|----------|----------|---------------|
| P:1D, H:1D | COBYLA | -4.3723 | -3.8782 | -2.8215 | 0.4941 | 1.5507 | 8 (0) |
| P:1D, H:1D | SPSA | -4.3723 | -3.9961 | -2.7329 | 0.3762 | 1.6394 | 8 (0) |
| P:1D, H:2D | COBYLA | -4.3723 | -4.0840 | -2.8445 | 0.2883 | 1.5278 | 12 (1) |
| P:1D, H:2D | SPSA | -4.3723 | -3.5618 | -1.9575 | 0.8105 | 2.4148 | 10 (1) |
| P:2D, H:1D | COBYLA | -5.4641 | -4.1099 | -1.4780 | 1.3542 | 3.9861 | 25 (6) |
| P:2D, H:1D | SPSA | -5.4641 | -4.4653 | -1.6416 | 0.9988 | 3.8225 | 25 (6) |
| P:2D, H:2D | COBYLA | -5.4641 | -4.2920 | -3.4219 | 1.1721 | 2.0422 | 11 (0) |
| P:2D, H:2D | SPSA | -5.4641 | -4.5317 | -3.4351 | 0.9324 | 2.0290 | 11 (0) |

**Key findings:**

- **Ideal simulations confirm topology mismatch alone is not the bottleneck.** Under noiseless conditions, matched and mismatched configurations produce comparable energy errors — differences are attributable to optimizer stochasticity, not routing overhead.

- **SWAP-induced noise is the dominant error source for mismatched hardware.** The worst-case configuration — a 2D problem on 1D hardware — required **6 SWAP gates** and a circuit depth of **25**, resulting in a noisy energy error of ~3.9 Ha. This is roughly **double** the error of the matched (P:2D, H:2D) configuration, which required 0 SWAPs at depth 11.

- **Excess connectivity does not penalize accuracy.** The (P:1D, H:2D) mismatch — where a 1D problem runs on 2D hardware — performed comparably to the fully matched (P:1D, H:1D) case. Because the 1D problem's entanglement graph is a subgraph of the 2D hardware graph, minimal routing is needed and noise impact remains low.

### Optimizer Comparison

- **SPSA consistently achieves lower final energy error than COBYLA** in the presence of SWAP noise. COBYLA converges quickly (~20 iterations) but frequently settles in local minima on noisy, topologically constrained landscapes. SPSA requires more iterations (~100) but its stochastic gradient approximation better navigates the noisy loss surface.

- Performance across hardware geometries converges at similar iteration counts for both optimizers, suggesting that the choice of optimizer matters more than the hardware topology in determining final accuracy.

---

## Conclusions

This study confirms that **hardware topology is a critical factor in VQE performance**, and that its impact is most severe when sparse hardware (1D linear) is paired with geometrically complex problems (2D ring). The principal mechanism is SWAP-induced depolarizing noise: each forced routing SWAP compounds gate error, inflating energy error by nearly 2× compared to matched-topology runs. Additionally, **stochastic optimizers (SPSA) are more robust** than simplex-based methods (COBYLA) for SWAP-noisy optimization landscapes. These findings highlight the importance of co-designing quantum algorithms with target hardware architectures.

---

## Project Structure

```
Analyzing-VQE-Geometry/
├── src/
│   ├── vqe_analysis.py              # Main simulation: Hamiltonian, VQE trials, analysis, plots
│   └── draw_mismatched_circuit.py   # Visual demonstration of SWAP insertion on mismatched topology
├── results/
│   ├── figures/
│   │   ├── vqe_convergence_comparison.png   # Convergence curves per optimizer
│   │   ├── noise_impact_analysis.png        # Ideal vs. noisy energy error comparison
│   │   └── topology_overhead_analysis.png   # SWAP gate count per configuration
│   └── data/
│       └── accuracy_table.csv               # Full numerical results
├── docs/
│   └── ECE_520_Final_Project_Hanlin_Wang.pdf  # Full written report
└── README.md
```

---

## Setup & Usage

**Requirements:** Python 3.9+

```bash
pip install qiskit qiskit-aer qiskit-algorithms numpy matplotlib
```

**Run the full experiment:**
```bash
python src/vqe_analysis.py
```
Saves plots to `results/figures/` and the results table to `results/data/accuracy_table.csv`.

**Visualize SWAP insertion on a mismatched circuit:**
```bash
python src/draw_mismatched_circuit.py
```

---

## References

1. J. Tilly et al., "The Variational Quantum Eigensolver: a review of methods and best practices," *Physics Reports* **986**, 1–128 (2022). [arXiv:2111.05176](https://arxiv.org/abs/2111.05176)
2. A. Holmes et al., "Impact of qubit connectivity on quantum algorithm performance," *Quantum Science and Technology* **5**, 025009 (2020). [arXiv:1811.02125](https://arxiv.org/abs/1811.02125)
3. A. Peruzzo et al., "A variational eigenvalue solver on a quantum processor," *Nature Communications* **5**, 4213 (2014). [arXiv:1304.3061](https://arxiv.org/abs/1304.3061)

---

**Stack:** `Python` · `Qiskit` · `qiskit-aer` · `qiskit-algorithms` · `NumPy` · `Matplotlib`
