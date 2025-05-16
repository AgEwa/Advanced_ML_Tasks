import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm


def g_func(x):
    return 4.26 * (np.exp(-x) - 4 * np.exp(-2 * x) + 3 * np.exp(-3 * x))


def generate_data(n, seed=42):
    if seed:
        np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 4, n))
    y_true = g_func(x)
    y_noisy = y_true + np.random.normal(0, 0.1, size=n)
    return x, y_noisy, y_true


def compute_mse(X, y_pred):
    n = len(X)
    y_true = g_func(X)
    return np.sum((y_true - y_pred) ** 2) / n


def nw_estimator(x_new, X, y, h, kernel=norm.pdf):
    x_new = np.atleast_1d(x_new)
    weights = np.array([[kernel((xi_new - X_i) / h) / h for X_i in X] for xi_new in x_new])

    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights
    return np.dot(weights, y)  # Weighted sum to get estimates


def run_plot_experiment(n, h=1.0, s=1.0, kernel=norm.pdf, seed=42, plot=True):
    X, y, y_true = generate_data(n, seed)
    x_plot = np.linspace(X.min(), X.max(), max(n, 100))

    y_nw = nw_estimator(x_plot, X, y, h, kernel)
    spline = UnivariateSpline(X, y, s=s)
    y_sp = spline(x_plot)
    mse_nw = compute_mse(x_plot, y_nw)
    mse_sp = compute_mse(x_plot, y_sp)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label="Observed Data", color="gray")
        plt.plot(x_plot, y_nw, label="Nadaraya-Watson", color="blue", linestyle="dashed")
        plt.plot(x_plot, y_sp, label="Spline", color="lime", linestyle="dashed")
        plt.plot(X, y_true, label="True Data", color="red")

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Comparison: Nadaraya-Watson vs. Splines.\n n={}, h={:.2f}, s={:.2f}".format(n, h, s))
        plt.show()
        print(f"Nadaraya-Watson MSE: {mse_nw:.4f}")
        print(f"Spline MSE: {mse_sp:.4f}")
    return mse_nw, mse_sp


def plot_mse(x, mse_nws, mse_sps, xlabel="Bandwidth h"):
    plt.figure(figsize=(10, 6))
    plt.plot(x, mse_nws, label="Nadaraya-Watson", color="blue", marker="o")
    plt.plot(x, mse_sps, label="Spline", color="lime", marker="o")
    plt.legend()
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel(xlabel)
    plt.ylabel("MSE")
    plt.title("MSE Comparison: Nadaraya-Watson vs. Splines")
    plt.show()


def find_best_bandwidth(n, h_range=None, kernel=norm.pdf, seed=42):
    best_mse = np.inf
    best_h = None
    patience = 3
    k = 0
    if h_range is None:
        h_range = np.linspace(0.01, 0.1, 20)

    X, y, y_true = generate_data(n, seed)
    x_plot = np.linspace(X.min(), X.max(), max(n, 100))

    for h in h_range:
        y_nw = nw_estimator(x_plot, X, y, h, kernel)
        mse_nw = compute_mse(x_plot, y_nw)
        if best_mse > mse_nw:
            best_mse = mse_nw
            best_h = h
            k=0
        elif k < patience:
            k = k + 1
        else:
            break

    return best_h

def find_best_smoothing(n, kernel=norm.pdf, seed=42):
    best_mse = np.inf
    best_s = None
    patience = 3
    k = 0
    if n<140:
        s_range = np.linspace(0.1, 1.5, 20)
    elif n<280:
        s_range=np.linspace(1.0, 3.0, 15)
    else:
        s_range = np.linspace(2.5, 4.2, 10)

    X, y, y_true = generate_data(n, seed)
    x_plot = np.linspace(X.min(), X.max(), max(n, 100))

    for s in s_range:
        spline = UnivariateSpline(X, y, s=s)
        y_sp = spline(x_plot)
        mse_sp = compute_mse(x_plot, y_sp)
        if best_mse > mse_sp:
            best_mse = mse_sp
            best_s = s
            k=0
        elif k < patience:
            k = k + 1
        else:
            break

    return best_s

def plot_params(x, hs, ss):
    plt.figure(figsize=(10, 6))
    plt.plot(x, hs, label="Nadaraya-Watson", color="blue", marker="o")
    plt.plot(x, ss, label="Spline", color="lime", marker="o")
    plt.legend()
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("Sample size")
    plt.ylabel("Parameter (s or h)")
    plt.title("Sample size influence on params: Nadaraya-Watson vs. Splines")
    plt.show()
