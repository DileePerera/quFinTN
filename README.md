# QuFinTN: Quantum-Inspired Tensor Networks for Portfolio Optimization

**A practical Python library for quantum-inspired portfolio optimization using Matrix Product States (MPS) and intelligent heuristics.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

QuFinTN is a **finance-first** library that implements quantum-inspired tensor network methods for portfolio optimization. It is inspired by the seminal paper:

> *Dynamic Portfolio Optimization with Real Datasets Using Quantum Processors and Quantum-Inspired Tensor Networks* (Mugel et al., 2022, arXiv:2007.00017)

This library allows you to:
- Compare **Classical Markowitz** optimization with **Quantum-Inspired MPS** methods
- Run walk-forward backtests
- Experiment with different bond dimensions and optimization strategies

---

## Features

- **MPS Optimizer** (Quantum-Inspired): Enhanced v3 with intelligent heuristics
- **Classical Markowitz** baseline using `cvxpy`
- **Robust data loader** with `yfinance`
- **Walk-forward backtesting** engine
- **Hamiltonian formulation** with risk, return, and transaction cost modeling
- Easy-to-use API with detailed results and visualizations

---

## Performance (Example Backtest)

**15 US stocks | 4 years of data | Walk-forward (1Y train, 3M test)**

| Method                    | Sharpe Ratio | CAGR    | Comment                     |
|--------------------------|--------------|---------|-----------------------------|
| **MPS Enhanced v3**      | **1.601**    | **28.0%** | **Best overall**            |
| Classical Markowitz      | 1.376        | 16.5%   | More concentrated           |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuFinTN.git
cd QuFinTN

# Install in editable mode
pip install -e .

#Requirements
pip install numpy pandas yfinance cvxpy tensornetwork quimb matplotlib plotly
```
---

### Key Components

* `MPSPortfolioOptimizer` — Main quantum-inspired optimizer
* `PortfolioHamiltonian` — QUBO-style formulation
* `PortfolioBacktester` — Walk-forward testing engine
* `MarkowitzOptimizer` — Classical baseline

---

### References

1. Samuel Mugel et al. (2022). Dynamic Portfolio Optimization with Real Datasets Using Quantum Processors and Quantum-Inspired Tensor Networks. arXiv:2007.00017
2. Tensor Network methods and DMRG-style optimization

---

### License

MIT License © 2026 Dileesha Perera
