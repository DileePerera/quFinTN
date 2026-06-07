

import numpy as np
from typing import Tuple, Optional, Dict
import warnings


class PortfolioHamiltonian:

    def __init__(self):
        pass

    def build(self,
              mu: np.ndarray,
              sigma: np.ndarray,
              budget: float = 1.0,
              lambda_risk: float = 1.0,
              transaction_cost: float = 0.001,
              prev_weights: Optional[np.ndarray] = None,
              penalty_strength: float = 10.0) -> Tuple[np.ndarray, Dict]:
        """
        Build a high-quality QUBO/Ising Hamiltonian for portfolio optimization.

        Returns:
            H_matrix: Quadratic Hamiltonian matrix
            metadata: Details about formulation
        """
        n = len(mu)

        # === 1. Risk Term (Quadratic) ===
        risk_term = lambda_risk * sigma.copy()

        # === 2. Return Term (Linear → Diagonal) ===
        return_term = np.diag(-mu)

        # === 3. Budget Constraint (Soft Penalty) ===
        # Penalty for sum(w) != budget
        budget_penalty = self._build_budget_penalty(n, budget, penalty_strength)

        # === 4. Transaction Costs (if previous weights given) ===
        transaction_term = self._build_transaction_cost_term(
            n, transaction_cost, prev_weights
        )

        # === Combine All Terms ===
        H = risk_term + return_term + budget_penalty + transaction_term

        metadata = {
            "n_assets": n,
            "budget": budget,
            "lambda_risk": lambda_risk,
            "transaction_cost": transaction_cost,
            "penalty_strength": penalty_strength,
            "formulation_type": "MeanVariance_QUBO",
            "is_sparsity": float(np.count_nonzero(H) / H.size)
        }

        print(f"Hamiltonian built for {n} assets | Risk weight: {lambda_risk} | "
              f"Transaction cost: {transaction_cost}")

        return H, metadata

    def _build_budget_penalty(self, 
                            n: int, 
                            budget: float, 
                            penalty: float) -> np.ndarray:
        """Soft constraint: (sum w_i - budget)^2"""
        H_penalty = penalty * np.ones((n, n))
        # Diagonal correction
        np.fill_diagonal(H_penalty, H_penalty.diagonal() - 2 * penalty * budget)
        return H_penalty

    def _build_transaction_cost_term(self,
                                   n: int,
                                   trans_cost: float,
                                   prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear transaction cost approximation"""
        if prev_weights is None:
            return np.zeros((n, n))

        # Approximate |w_i - prev_w_i| as quadratic term (common trick)
        trans_term = np.zeros((n, n))
        for i in range(n):
            trans_term[i, i] += trans_cost * abs(1 - 2 * prev_weights[i])

        return trans_term

    def to_ising(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert QUBO Hamiltonian to Ising form (for some TN solvers)
        H = sum_{i<j} J_{ij} Z_i Z_j + sum_i h_i Z_i + const
        """
        n = H.shape[0]
        J = H.copy() / 4.0
        np.fill_diagonal(J, 0)

        h = np.zeros(n)
        for i in range(n):
            h[i] = np.sum(H[i]) / 2.0

        return J, h

    def scale_hamiltonian(self, H: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """Scale Hamiltonian to improve numerical stability"""
        return H * scale_factor / np.max(np.abs(H))


# ================ Convenient Wrapper Function ================

def build_portfolio_hamiltonian(mu: np.ndarray,
                               sigma: np.ndarray,
                               budget: float = 1.0,
                               lambda_risk: float = 1.0,
                               transaction_cost: float = 0.001,
                               prev_weights: Optional[np.ndarray] = None,
                               penalty_strength: float = 15.0) -> Tuple[np.ndarray, Dict]:
    """
    High-level function to build Hamiltonian (recommended to use)
    """
    builder = PortfolioHamiltonian()
    H, metadata = builder.build(
        mu=mu,
        sigma=sigma,
        budget=budget,
        lambda_risk=lambda_risk,
        transaction_cost=transaction_cost,
        prev_weights=prev_weights,
        penalty_strength=penalty_strength
    )
    return H, metadata