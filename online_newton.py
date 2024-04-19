# Copyright: Xiaobo @ https://github.com/xiaobo-yang
# Description: Online Newton Algorithm for Portfolio Optimization
# Reference: Introduction to Online Convex Optimization, Elad Hazan, 2016

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple, List
from tqdm import tqdm

def projection(
    x: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    Projection x onto the probability simplex with the matrix-induced norm ||x||_A = sqrt(x^T A x).
    Input:
        x: np.array of shape (n,)
        A: np.array of shape (n, n)
    
    Output:
        result: np.array of shape (n,) that minimizes ||p - x||_A subject to p >= 0 and sum(p) = 1
    """
    n = len(x)
    objective = lambda p: np.dot((p - x).T, np.dot(A, (p - x)))
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1}, 
                   {'type': 'ineq', 'fun': lambda p: p}) 
    initial_guess = np.ones(n) / n  
    result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

    return result.x

def online_newton_portfolio(
    n: int, 
    T: int, 
    ratios: np.ndarray, 
    epsilon: Optional[float] = 1e-3,
    use_eps: Optional[bool] = False,
):
    """
    Online Newton Algorithm for Portfolio Optimization.
    
    Input:
        n: number of assets
        T: time horizon
        ratios: np.array of shape (n, T) representing the price change ratios of n assets at each time step
        epsilon: regularization parameter
    Output:
        x_iter: portfolio weights of each iter
        r_iter: price change ratios of each iter
        cost_iter: cost of each iter
    """
    G, D = np.sqrt(n) * 1.1 / 0.9, 2  
    # G is an upper bound on the l2 norm of the gradient of the loss function. 
    # As it is the ratio of price changes, a rough estimate of the bound can be made, with an estimated range of 0.9-1.1; 
    # And D is the radius of the probability simplex, which makes it easy to estimate the upper limit of an l2 norm to be 2

    gamma = min(1, 1/(G*D)) / 2  
    # alpha=1, alpha is a parameter of alpha exp concave

    if use_eps:
        epsilon = 1 / (gamma * D) ** 2  
        # The theoretical guarantee of convergence in the book requires epsilon, which may be very large. 
        # Experiments have found that it is not good to choose that method, and using a very small epsilon is actually better. 
        # In fact, looking at the proof of Lemma 4.6, the smaller the epsilon, the smaller the reget bound. 
        # The method of taking this epsilon is more like reaching the end of the entire Theorem 4.5 for the sake of mathematical aesthetics

    A = epsilon * np.eye(n)
    x = np.ones(n) / n
    x_iter, r_iter, cost_iter = np.zeros((n, T)), np.zeros((n, T)), np.zeros(T)

    for t in range(T):
        r = ratios[:,t]
        cost = -np.log(np.dot(r, x))
        grad = -r / np.dot(r, x)
        A += np.outer(grad, grad)
        y = x - np.linalg.inv(A) @ grad / gamma
        x = projection(y, A)
        x_iter[:,t], r_iter[:,t], cost_iter[t] = x, r, cost # Note that the cost here is the cost of x at time t-1, not the cost of x at time t!

    return x_iter, r_iter, cost_iter

def flh_portfolio(
    n: int, 
    T: int, 
    ratios: np.ndarray, 
    epsilon: Optional[float] = 1e-3, 
    use_eps: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[np.ndarray]]:
    """
    Follow the Leading History Algorithm for Portfolio Optimization.
    
    Input:
        n: number of assets
        T: time horizon
        ratios: np.array of shape (n, T) representing the price change ratios of n assets at each time step
        epsilon: regularization parameter
        use_eps: flag indicating whether to use epsilon
    
    Output:
        all_xs: portfolio weights of each iter
        all_costs: cost of each iter
        all_ps: expert weights at each iter
        all_tmp_xs: temporary portfolio weights at each iter
    """
    all_costs = np.zeros(T)
    all_xs = np.zeros((n, T))
    all_tmp_xs = []
    tmp_x = np.ones(n) / n
    p = [1]
    all_ps = [p]

    for t in tqdm(range(1, T + 1)):
        tmp_r = ratios[:, t - 1]
        tmp_cost = -np.log(np.dot(tmp_r, tmp_x))
        all_costs[t - 1] = tmp_cost

        # Online Newton steps
        xs = np.zeros((n, t))
        for j in range(t):
            tmp_ratios = ratios[:, j:t]
            x, _, _ = online_newton_portfolio(n, t - j, tmp_ratios, epsilon, use_eps=use_eps)
            xs[:, j] = x[:, -1]
        tmp_x = xs @ np.array(p)  # Average portfolio allocation
        all_xs[:, t - 1] = tmp_x
        all_tmp_xs.append(xs)

        # Receive new f_t, update expert weights
        if t == T:
            break
        new_r = ratios[:, t]
        exp_costs = (new_r @ xs) * np.array(p)
        for i in range(t):
            p[i] = (1 - 1 / (t + 1)) * exp_costs[i] / exp_costs.sum()
        p.append(1 / (t + 1))
        all_ps.append(p)

    return all_xs, all_costs, all_ps, all_tmp_xs  # Note that the cost here is the cost of x at time t-1, not the cost of x at time t!

def flh2_portfolio(
    n: int, 
    T: int, 
    ratios: np.ndarray, 
    epsilon: Optional[float] = 1e-3, 
    use_eps: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[np.ndarray]]:
    """
    Follow the Leading History Algorithm 2 for Portfolio Optimization.
    
    Input:
        n: number of assets
        T: time horizon
        ratios: np.array of shape (n, T) representing the price change ratios of n assets at each time step
        epsilon: regularization parameter
        use_eps: flag indicating whether to use epsilon
    
    Output:
        all_xs: portfolio weights of each iter
        all_costs: cost of each iter
        all_ps: expert weights at each iter
        all_tmp_xs: temporary portfolio weights at each iter
    """
    all_costs = np.zeros(T)
    all_xs = np.zeros((n, T))
    all_tmp_xs = []
    tmp_x = np.ones(n) / n
    p = np.array([1.] + [0. for _ in range(T - 1)])
    all_ps = [p]
    S_t = [0]

    for t in tqdm(range(1, T + 1)):
        tmp_r = ratios[:, t - 1]
        tmp_cost = -np.log(np.dot(tmp_r, tmp_x))
        all_costs[t - 1] = tmp_cost

        # Online Newton steps
        xs = np.zeros((n, len(S_t)))
        for id in range(len(S_t)):
            j = S_t[id]
            tmp_ratios = ratios[:, j:t]
            x, _, _ = online_newton_portfolio(n, t - j, tmp_ratios, epsilon, use_eps=use_eps)
            xs[:, id] = x[:, -1]
        tmp_x = xs @ p[S_t]  # Average portfolio allocation
        all_xs[:, t - 1] = tmp_x
        all_tmp_xs.append(xs)

        # Receive new f_t, update expert weights
        if t == T:
            break
        new_r = ratios[:, t]
        exp_costs = (new_r @ xs) * p[S_t]
        for id in range(len(S_t)):
            i = S_t[id]
            p[i] = exp_costs[id] / exp_costs.sum()
        p[t] = 1 / t

        # Pruning
        for i in S_t:
            r, k = i, 0
            while r != 0 and r % 2 == 0:
                r = r // 2
                k += 1
            if t - i > 2 ** (k + 2):
                S_t.remove(i)
        S_t.append(t)

        # New expert weights
        norm_const = p[S_t].sum()
        for i in S_t:
            p[i] = p[i] / norm_const
        all_ps.append(p)

    return all_xs, all_costs, all_ps, all_tmp_xs  # Note that the cost here is the cost of x at time t-1, not the cost of x at time t!