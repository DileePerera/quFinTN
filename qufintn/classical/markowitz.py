import cvxpy as cp
import numpy as np

class MarkowitzOptimizer:
    def optimize(self, mu, sigma, target_return=None):
        n = len(mu)
        w = cp.Variable(n)
        risk = cp.quad_form(w, sigma)
        
        constraints = [cp.sum(w) == 1, w >= 0]
        if target_return:
            constraints.append(w.T @ mu >= target_return)
        
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()
        
        return w.value