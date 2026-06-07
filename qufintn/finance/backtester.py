

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional


class PortfolioBacktester:
    """
    Advanced backtester supporting MPS vs Classical comparison
    """
    
    def __init__(self, risk_free_rate: float = 0.03, rebalance_frequency: str = 'M'):
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = rebalance_frequency
        self.results = {}           # MPS results
        self.classical_results = {} # Classical results
        
    def run_walk_forward(self, 
                        returns: pd.DataFrame,
                        optimizer,
                        train_window: int = 252,
                        test_window: int = 63,
                        lambda_risk: float = 1.0) -> Dict:
        """
        Perform walk-forward backtesting
        """
        n_periods = len(returns)
        portfolio_values = [1.0]
        dates = returns.index
        weights_history = []
        
        print(f"Running walk-forward backtest | Train: {train_window} | Test: {test_window} days")
        
        i = train_window
        while i + test_window <= n_periods:
            train_data = returns.iloc[i-train_window:i]
            test_data = returns.iloc[i:i+test_window]
            
            mu = train_data.mean() * 252
            sigma = train_data.cov() * 252
            
            try:
                if hasattr(optimizer, 'optimize'):
                    result = optimizer.optimize(
                        Q=sigma.values,
                        q=mu.values,
                        budget=1.0,
                        lambda_risk=lambda_risk
                    )
                    weights = result['weights']
                else:
                    weights = optimizer.optimize(mu.values, sigma.values)
            except:
                weights = np.ones(len(mu)) / len(mu)
            
            weights_history.append((dates[i], weights))
            
            period_returns = test_data @ weights
            portfolio_values.extend(portfolio_values[-1] * (1 + period_returns).cumprod())
            
            i += test_window
        
        final_results = {
            'portfolio_values': pd.Series(portfolio_values, index=dates[:len(portfolio_values)]),
            'weights_history': weights_history,
            'total_return': portfolio_values[-1] - 1,
            'cagr': self._calculate_cagr(portfolio_values[-1], len(returns)),
            'sharpe_ratio': self._calculate_sharpe(pd.Series(portfolio_values).pct_change().dropna())
        }
        
        return final_results
    
    def compare(self, 
               returns: pd.DataFrame,
               mps_optimizer,
               classical_optimizer,
               train_window: int = 252,
               test_window: int = 63) -> Dict:
        """Run comparison and store both results"""
        print("=== Running MPS Optimizer Backtest ===")
        mps_results = self.run_walk_forward(returns, mps_optimizer, 
                                          train_window=train_window, 
                                          test_window=test_window)
        self.results = mps_results
        
        print("\n=== Running Classical Markowitz Backtest ===")
        classical_results = self.run_walk_forward(returns, classical_optimizer, 
                                                train_window=train_window, 
                                                test_window=test_window)
        self.classical_results = classical_results
        
        comparison = {
            'mps': {
                'total_return': mps_results['total_return'],
                'cagr': mps_results['cagr'],
                'sharpe': mps_results['sharpe_ratio']
            },
            'classical': {
                'total_return': classical_results['total_return'],
                'cagr': classical_results['cagr'],
                'sharpe': classical_results['sharpe_ratio']
            }
        }
        
        print("\n=== Final Comparison ===")
        print(f"MPS Optimizer     → Sharpe: {comparison['mps']['sharpe']:.3f} | CAGR: {comparison['mps']['cagr']:.1%}")
        print(f"Classical Markowitz → Sharpe: {comparison['classical']['sharpe']:.3f} | CAGR: {comparison['classical']['cagr']:.1%}")
        
        return comparison
    
    def plot_performance(self, title: str = "Quantum-Inspired MPS vs Classical Markowitz Performance"):
        """Plot both MPS and Classical curves on the same graph"""
        if 'portfolio_values' not in self.results:
            print("No results to plot.")
            return
        
        fig = go.Figure()
        
        # MPS Portfolio
        fig.add_trace(go.Scatter(
            x=self.results['portfolio_values'].index,
            y=self.results['portfolio_values'],
            mode='lines',
            name='MPS Optimizer (Quantum-Inspired)',
            line=dict(color='blue', width=3)
        ))
        
        # Classical Markowitz
        if self.classical_results and 'portfolio_values' in self.classical_results:
            fig.add_trace(go.Scatter(
                x=self.classical_results['portfolio_values'].index,
                y=self.classical_results['portfolio_values'],
                mode='lines',
                name='Classical Markowitz',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($1 Initial)",
            template="plotly_white",
            height=650,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )
        
        fig.show()
    
    # ====================== Helper Metrics ======================
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0.0
    
    def _calculate_cagr(self, final_value: float, n_days: int) -> float:
        years = n_days / 252.0
        return final_value ** (1 / years) - 1 if years > 0 else 0.0


# ================ Convenience Function ================

def run_full_backtest(returns: pd.DataFrame, 
                     mps_optimizer,
                     classical_optimizer=None,
                     train_window: int = 252,
                     test_window: int = 63):
    """High-level function"""
    backtester = PortfolioBacktester()
    
    if classical_optimizer is None:
        from qufintn.classical.markowitz import MarkowitzOptimizer
        classical_optimizer = MarkowitzOptimizer()
    
    comparison = backtester.compare(
        returns=returns,
        mps_optimizer=mps_optimizer,
        classical_optimizer=classical_optimizer,
        train_window=train_window,
        test_window=test_window
    )
    
    backtester.plot_performance()
    
    return backtester, comparison