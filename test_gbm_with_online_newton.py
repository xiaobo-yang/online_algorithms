import argparse
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from online_newton import online_newton_portfolio, flh_portfolio, flh2_portfolio

# Simulation data
def geometric_brownian_motion(T, N, mu, sigma, S0):
    """
    Generate geometric brownian motion.
    
    Input:
        T: float, time horizon
        N: int, number of time steps
        mu: float, drift
        sigma: float, volatility
        S0: float, initial stock price
        
    Output:
        t: np.array, time points
        S: np.array, stock prices
    """
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W 
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return t, S

def geometric_brownian_motion_changing(T, N, mu_func, sigma_func, S0):
    """
    Generate geometric brownian motion with changing parameters over time.
    
    Input:
        T: float, time horizon
        N: int, number of time steps
        mu_func: function, function defining the drift over time
        sigma_func: function, function defining the volatility over time
        S0: float, initial stock price
        
    Output:
        t: np.array, time points
        S: np.array, stock prices
    """
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    
    mu = mu_func(t)
    sigma = sigma_func(t)
    
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return t, S

# Example of varying mu and sigma over time
def mu_func(t):
    return 0.05 + 0.01 * np.sin(2 * np.pi * t / 65)

def sigma_func(t):
    return 0.2 + 0.15 * np.cos(2 * np.pi * t / 65)

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(description="Simulate geometric brownian motion")
    parser.add_argument("--T", type=float, default=1, help="Time horizon")
    parser.add_argument("--N", type=int, default=301, help="Number of time steps")
    parser.add_argument("--S0", type=float, default=100, help="Initial stock price")
    parser.add_argument("--simulations", type=int, default=5, help="Number of simulations")
    parser.add_argument("--use_flh", action='store_true', help="If use full time FLH")
    args = parser.parse_args()

    # Generate geometric brownian motion
    plt.figure(figsize=(27, 6))
    value_changes = [] 
    paths = []

    # Generate geometric brownian motion
    for i in range(args.simulations):
        # t, S = geometric_brownian_motion(args.T, args.N, mu, sigma, args.S0)
        t, S = geometric_brownian_motion_changing(args.T, args.N, mu_func, sigma_func, args.S0)
        plt.plot(t, S)
        value_changes.append(S[1:] / S[:-1])
        paths.append(S)

    value_changes = np.array(value_changes)
    paths = np.array(paths)

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Geometric Brownian Motion Simulations')
    plt.grid(True)
    plt.show(block=True)

    # solve max gain of fixed strategy
    x = cp.Variable(args.simulations)  
    objective = cp.Minimize(-cp.sum(cp.log(value_changes.T @ x)))
    constraints = [x >= 0, cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_x = x.value
    max_gain_costs = -np.log(value_changes.T @ optimal_x)
    print(f"Optimal solution: {optimal_x}")

    # solve online newton strategy
    xs, rs, costs = online_newton_portfolio(args.simulations, args.N-1, value_changes, use_eps=True)
    if args.use_flh:
        xs1, costs1, all_ps1, all_tmp_xs1 = flh_portfolio(args.simulations, args.N-1, value_changes, use_eps=True)
    xs2, costs2, all_ps2, all_tmp_xs2 = flh2_portfolio(args.simulations, args.N-1, value_changes, use_eps=True)

    plt.figure(figsize=(27, 6)) 
    plt.plot(np.linspace(0, 1, args.N)[1:], np.cumsum(costs),label='Online Newton Method')
    if args.use_flh:
        plt.plot(np.linspace(0, 1, args.N)[1:], np.cumsum(costs1),label='Online Newton Method with FLH')
    plt.plot(np.linspace(0, 1, args.N)[1:], np.cumsum(costs2),label='Online Newton Method with FLH2')
    plt.plot(np.linspace(0, 1, args.N)[1:], np.cumsum(max_gain_costs),label='max gain')
    plt.xlabel('Time')
    plt.ylabel('Cumulative loss')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

    plt.figure(figsize=(27, 6)) 
    plt.plot(np.linspace(0, 1, args.N)[1:], np.exp(-np.cumsum(costs)),label='Online Newton Method')
    if args.use_flh:
        plt.plot(np.linspace(0, 1, args.N)[1:], np.exp(-np.cumsum(costs1)),label='Online Newton Method with FLH')
    plt.plot(np.linspace(0, 1, args.N)[1:], np.exp(-np.cumsum(costs2)),label='Online Newton Method with FLH2')
    plt.plot(np.linspace(0, 1, args.N)[1:], np.exp(-np.cumsum(max_gain_costs)),label='max gain')
    plt.xlabel('Time')
    plt.ylabel('Money change(per dollars)')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
