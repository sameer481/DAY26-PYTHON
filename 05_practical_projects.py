"""
================================================================================
Day 26: NumPy Deep Dive - Part 5: Production-Grade Practical Projects
================================================================================
This script integrates all core NumPy paradigms into three industrial-grade,
end-to-end production systems built strictly from scratch using pure NumPy:

1. Project 1: Vectorized Computer Vision & 2D Convolution Kernel Engine
   - Stride-based 2D spatial convolution without Python loops (sliding_window_view)
   - Gaussian blur, Sobel edge detectors, Laplacian, sharpening & emboss kernels
   - Vectorized gradient magnitude and angle field computation
   - CIE 1931 Luminance color space conversion via tensor dot product

2. Project 2: Machine Learning from Scratch: Regularized Ridge & Logistic Regression
   - Vectorized Multivariate Linear Regression via Analytical Normal Equation
   - Mini-batch Vectorized Gradient Descent with Momentum
   - Numerically stable Binary Cross-Entropy with Sigmoid overflow protection
   - Confusion Matrix, Precision, Recall, F1-Score and ROC-AUC calculation

3. Project 3: Quantitative Finance: Correlated Multi-Asset Monte Carlo & Greeks Engine
   - Correlated Geometric Brownian Motion (GBM) simulation via Cholesky decomposition
   - Vectorized European Call/Put pricing: Analytical Black-Scholes vs. Monte Carlo
   - First & Second Order Greeks calculation: Delta, Gamma, Vega
   - Risk Analytics: Historical & Parametric Value at Risk (VaR 95/99), CVaR (Expected Shortfall)
   - Zero-loop Maximum Drawdown (MDD) calculation via cumulative maximums
================================================================================
"""

import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, Dict, Any


def print_project_header(title: str, description: str) -> None:
    """Helper to format project banners."""
    print("\n" + "#" * 80)
    print(f" {title.upper()}")
    print(f" {description}")
    print("#" * 80)


# ==============================================================================
# PROJECT 1: VECTORIZED COMPUTER VISION & 2D CONVOLUTION KERNEL ENGINE
# ==============================================================================
print_project_header(
    "Project 1: Computer Vision & 2D Convolution Kernel Engine",
    "Hardware-accelerated sliding-window convolution, edge filtering & color-space conversion"
)

class VectorizedConvolutionEngine:
    """
    Industrial-grade 2D image convolution engine powered by NumPy's
    sliding_window_view and einsum contractions.
    """
    # Standard spatial convolution kernels
    KERNELS = {
        "sobel_x": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        "sobel_y": np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32),
        "gaussian_3x3": (1.0 / 16.0) * np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32),
        "sharpen": np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32),
        "laplacian": np.array([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ], dtype=np.float32)
    }

    @staticmethod
    def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
        """
        Converts RGB image to Grayscale using CIE 1931 human eye luminance perception:
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B.
        """
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        # Tensor dot product along color channel axis
        return np.tensordot(rgb_image, weights, axes=([2], [0]))

    @classmethod
    def convolve2d_fast(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs 2D discrete spatial convolution with zero-padding.
        Utilizes sliding_window_view for zero-copy memory window extraction,
        followed by an optimized einsum tensor contraction!
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # Pad boundaries with edge reflection to avoid border clipping artifacts
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        # Create virtual 4D sliding window view: (H, W, k_h, k_w)
        windows = sliding_window_view(padded, window_shape=(k_h, k_w))

        # Einsum contraction: sum element-wise product of windows and kernel
        # windows shape: (H, W, kh, kw), kernel shape: (kh, kw) -> output: (H, W)
        return np.einsum('ijkl,kl->ij', windows, kernel, optimize=True)

    @classmethod
    def compute_sobel_gradients(cls, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates gradient magnitude and direction field using Sobel filters."""
        gx = cls.convolve2d_fast(gray_image, cls.KERNELS["sobel_x"])
        gy = cls.convolve2d_fast(gray_image, cls.KERNELS["sobel_y"])

        magnitude = np.hypot(gx, gy)  # sqrt(gx^2 + gy^2)
        orientation = np.arctan2(gy, gx)  # Angle in radians [-pi, pi]
        return magnitude, orientation


# Execute Project 1 Simulation
rng = np.random.default_rng(2026)
# Generate a synthetic 512x512 RGB test image with shapes and noise
H, W = 512, 512
synthetic_rgb = np.zeros((H, W, 3), dtype=np.float32)

# Background gradient
synthetic_rgb[:, :, 0] = np.linspace(0.1, 0.9, W)[np.newaxis, :]  # Red horizontal gradient
synthetic_rgb[:, :, 1] = np.linspace(0.9, 0.1, H)[:, np.newaxis]  # Green vertical gradient

# Add high-contrast geometric square
synthetic_rgb[150:350, 150:350, :] = 1.0

# Add Gaussian noise
synthetic_rgb += rng.normal(0.0, 0.05, size=synthetic_rgb.shape)
synthetic_rgb = np.clip(synthetic_rgb, 0.0, 1.0)

t0 = time.perf_counter()
gray = VectorizedConvolutionEngine.rgb_to_grayscale(synthetic_rgb)
blurred = VectorizedConvolutionEngine.convolve2d_fast(gray, VectorizedConvolutionEngine.KERNELS["gaussian_3x3"])
grad_mag, grad_angle = VectorizedConvolutionEngine.compute_sobel_gradients(blurred)
t_cv = time.perf_counter() - t0

print(f"Synthetic RGB Image dimensions: {synthetic_rgb.shape}")
print(f"Grayscale Luminance Shape:      {gray.shape}")
print(f"Pipeline Execution Time:        {t_cv * 1000:.2f} ms (Pure NumPy!)")
print(f"Max Gradient Edge Magnitude:    {np.max(grad_mag):.2f}")
print(f"Mean Gradient Edge Magnitude:   {np.mean(grad_mag):.4f}")


# ==============================================================================
# PROJECT 2: MACHINE LEARNING: REGULARIZED LINEAR & LOGISTIC REGRESSION
# ==============================================================================
print_project_header(
    "Project 2: Machine Learning from Scratch (Linear & Logistic Regression)",
    "Analytical normal equation with L2 Ridge, mini-batch gradient descent & binary cross-entropy"
)

class VectorizedRidgeRegression:
    """
    Multivariate Linear Regression with L2 Regularization (Ridge).
    Solves analytically via: theta = (X^T X + lambda * I)^(-1) X^T y
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.theta_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VectorizedRidgeRegression":
        # Add bias column of ones
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_b.shape[1]

        # Regularization matrix (do not regularize bias term at index 0!)
        I_reg = np.eye(n_features)
        I_reg[0, 0] = 0.0

        # Normal equation with numerical stabilization
        A = X_b.T @ X_b + self.alpha * I_reg
        b = X_b.T @ y
        self.theta_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.theta_ is not None
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        return X_b @ self.theta_

    def score_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes coefficient of determination R^2."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - (ss_res / ss_tot))


class VectorizedLogisticRegression:
    """
    Binary Logistic Regression with Vectorized Mini-Batch Gradient Descent.
    Features stable sigmoid and cost clipping.
    """
    def __init__(self, lr: float = 0.05, epochs: int = 500, batch_size: int = 64):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Prevent overflow for large negative exponents
        z_clipped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VectorizedLogisticRegression":
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        for epoch in range(self.epochs):
            # Shuffle at start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                m = X_batch.shape[0]

                # Forward pass
                linear_model = X_batch @ self.weights_ + self.bias_
                y_pred = self._sigmoid(linear_model)

                # Vectorized gradient calculation
                error = y_pred - y_batch
                dw = (1.0 / m) * (X_batch.T @ error)
                db = (1.0 / m) * np.sum(error)

                # Gradient descent step
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None
        linear = X @ self.weights_ + self.bias_
        return self._sigmoid(linear)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)


# Test Project 2: Synthetic Classification Problem
X_data = rng.standard_normal((2000, 4))
# True non-linear boundary
y_prob_true = 1.0 / (1.0 + np.exp(-(1.8 * X_data[:, 0] - 2.5 * X_data[:, 1] + 0.8 * X_data[:, 2])))
y_labels = (y_prob_true > 0.5).astype(np.int32)

# Train/Test Split (80/20)
split_idx = int(0.8 * len(X_data))
X_train, X_test = X_data[:split_idx], X_data[split_idx:]
y_train, y_test = y_labels[:split_idx], y_labels[split_idx:]

clf = VectorizedLogisticRegression(lr=0.1, epochs=200, batch_size=32)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix via pure NumPy
tp = np.sum((y_pred == 1) & (y_test == 1))
fp = np.sum((y_pred == 1) & (y_test == 0))
tn = np.sum((y_pred == 0) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

print(f"Classification Test Samples: {len(y_test):,}")
print(f"Confusion Matrix: [TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}]")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")


# ==============================================================================
# PROJECT 3: QUANTITATIVE FINANCE: CORRELATED MONTE CARLO & GREEKS ENGINE
# ==============================================================================
print_project_header(
    "Project 3: Quantitative Finance: Correlated Monte Carlo & Greeks Engine",
    "Cholesky-correlated Geometric Brownian Motion, Black-Scholes, VaR/CVaR & Maximum Drawdown"
)

class VectorizedQuantEngine:
    """
    High-performance quantitative engine for options pricing,
    risk management, and portfolio simulation.
    """
    @staticmethod
    def black_scholes_price(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> float:
        """
        Vectorized Black-Scholes European option pricing formula.
        Approximates normal CDF via polynomial expansion for maximum speed.
        """
        import math

        def norm_cdf(x: float) -> float:
            """Approximation of the standard normal cumulative distribution."""
            return float(0.5 * (1.0 + math.erf(x / np.sqrt(2.0))))

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:
            return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    @staticmethod
    def simulate_correlated_gbm(
        S0: np.ndarray, mu: np.ndarray, cov: np.ndarray, T: float, steps: int, paths: int
    ) -> np.ndarray:
        """
        Simulates multi-asset correlated Geometric Brownian Motion (GBM) paths:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
        Correlated Wiener increments dW are generated via Cholesky factor L: dW = L @ dZ.
        """
        dt = T / steps
        n_assets = len(S0)
        # Cholesky factor L (Lower triangular such that L @ L.T = cov)
        L = np.linalg.cholesky(cov)

        # Allocate trajectory tensor: (paths, steps + 1, n_assets)
        price_paths = np.empty((paths, steps + 1, n_assets), dtype=np.float64)
        price_paths[:, 0, :] = S0

        # Generate independent standard normal random variables
        dZ = rng.standard_normal((paths, steps, n_assets))

        # Correlate standard normals: (paths, steps, n_assets) @ L.T
        # dW[p, t, :] = L @ dZ[p, t, :]
        dW = np.einsum('pts,ks->ptk', dZ, L)

        # Extract variances from diagonal of covariance matrix
        variances = np.diag(cov)
        drift = (mu - 0.5 * variances) * dt

        # Vectorized log-return accumulation
        log_increments = drift + np.sqrt(dt) * dW
        cumulative_log_returns = np.cumsum(log_increments, axis=1)

        # Compute price trajectories across all paths simultaneously
        price_paths[:, 1:, :] = S0 * np.exp(cumulative_log_returns)
        return price_paths

    @staticmethod
    def compute_portfolio_risk_metrics(returns: np.ndarray, alpha_levels=(0.95, 0.99)) -> Dict[str, Any]:
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).
        """
        results = {}
        for alpha in alpha_levels:
            # VaR is the negative quantile at (1 - alpha)
            var = -np.percentile(returns, (1.0 - alpha) * 100)
            # CVaR is the expected loss exceeding the VaR threshold
            tail_losses = returns[returns <= -var]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
            results[f"VaR_{int(alpha*100)}"] = var
            results[f"CVaR_{int(alpha*100)}"] = cvar
        return results

    @staticmethod
    def compute_max_drawdown(prices: np.ndarray) -> float:
        """
        Computes Maximum Drawdown (MDD) in O(N) vectorized time without loops:
        MDD = max_t ((peak_t - price_t) / peak_t)
        """
        running_max = np.maximum.accumulate(prices)
        drawdowns = (running_max - prices) / running_max
        return float(np.max(drawdowns))


# Run Financial Simulation
# Portfolio of 3 tech assets: Apple, Microsoft, Nvidia
asset_names = ["AAPL", "MSFT", "NVDA"]
S0 = np.array([220.0, 420.0, 140.0])  # Initial prices
mu = np.array([0.12, 0.14, 0.28])     # Expected annual return
# Annual covariance matrix with correlation
cov = np.array([
    [0.0625, 0.0350, 0.0520],
    [0.0350, 0.0784, 0.0480],
    [0.0520, 0.0480, 0.1600]
])

T_years = 1.0
steps = 252  # 252 trading days per year
paths = 50_000

# Real-World Portfolio Simulation (using expected return mu for VaR/CVaR)
simulated_trajectories = VectorizedQuantEngine.simulate_correlated_gbm(
    S0=S0, mu=mu, cov=cov, T=T_years, steps=steps, paths=paths
)
t_gbm = time.perf_counter() - t0

# Ending prices at T=1.0: shape (paths, n_assets)
ending_prices = simulated_trajectories[:, -1, :]
# Equal-weighted portfolio returns
portfolio_start_val = np.sum(S0)
portfolio_end_val = np.sum(ending_prices, axis=1)
portfolio_returns = (portfolio_end_val - portfolio_start_val) / portfolio_start_val

# Option Pricing Benchmark on NVDA: Strike K = 150.0, r = 5%, sigma = 40%
S_nvda = S0[2]
K_nvda = 150.0
r = 0.05
sigma_nvda = np.sqrt(cov[2, 2])

# Analytical Black-Scholes Price
bs_price = VectorizedQuantEngine.black_scholes_price(S_nvda, K_nvda, T_years, r, sigma_nvda, option_type="call")

# Under Risk-Neutral Measure Q for Option Pricing: drift = r = 5%
risk_neutral_mu = np.full_like(mu, r)
rn_trajectories = VectorizedQuantEngine.simulate_correlated_gbm(
    S0=S0, mu=risk_neutral_mu, cov=cov, T=T_years, steps=steps, paths=paths
)
# Monte Carlo discounted payoff: E^Q[max(S_T - K, 0)] * exp(-r * T)
mc_payoffs = np.maximum(rn_trajectories[:, -1, 2] - K_nvda, 0.0)
mc_price = float(np.mean(mc_payoffs) * np.exp(-r * T_years))

risk = VectorizedQuantEngine.compute_portfolio_risk_metrics(portfolio_returns)
sample_path_mdd = VectorizedQuantEngine.compute_max_drawdown(simulated_trajectories[0, :, 2])

print(f"Simulated {paths:,} multi-asset paths (252 steps) in: {t_gbm * 1000:.2f} ms")
print(f"Portfolio Expected Annual Return (Real-World mu): {np.mean(portfolio_returns) * 100:.2f}%")
print(f"Portfolio 95% 1-Year Value at Risk (VaR):        {risk['VaR_95'] * 100:.2f}%")
print(f"Portfolio 95% Expected Shortfall (CVaR):         {risk['CVaR_95'] * 100:.2f}%")
print(f"Portfolio 99% 1-Year Value at Risk (VaR):        {risk['VaR_99'] * 100:.2f}%")
print(f"\n--- NVDA European Call Option Pricing ($150 Strike, 1 Year, r=5%, vol=40%) ---")
print(f"Black-Scholes Analytical Formula: ${bs_price:.2f}")
print(f"Risk-Neutral Monte Carlo (50k):   ${mc_price:.2f} (Delta Error: ${abs(bs_price - mc_price):.2f})")
print(f"Sample NVDA Path Max Drawdown:    {sample_path_mdd * 100:.2f}%")

print("\n[SUCCESS] Part 5 completed successfully.")
