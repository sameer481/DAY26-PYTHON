"""
================================================================================
Day 26: NumPy Deep Dive - Part 3: Linear Algebra, Decompositions & Einstein Sum
================================================================================
Linear Algebra is the mathematical language of machine learning, quantum
mechanics, graphics, and quantitative finance. Under the hood, NumPy delegates
all operations in `np.linalg` to highly tuned low-level BLAS and LAPACK
libraries (such as OpenBLAS, Apple Accelerate, or Intel MKL).

Theoretical Deep-Dive:
----------------------
1. Matrix Multiplication Semantics:
   - `*` operator: Element-wise Hadamard product (shape must be broadcastable).
   - `@` operator / `np.matmul()`: Strict matrix multiplication adhering to
     linear algebra rules. Handles batch dimensions `(B, N, K) @ (B, K, M) -> (B, N, M)`.
   - `np.dot()`: Multiplies along the last axis of A and the second-to-last axis of B.
     For 2D matrices it equals `matmul`, but behaves differently on tensors!

2. Numerical Stability: Why `np.linalg.inv()` is an Anti-Pattern:
   To solve Ax = b, beginners often compute `x = np.linalg.inv(A) @ b`.
   Computing the explicit inverse requires O(2N^3) floating point operations,
   amplifies machine rounding errors, and suffers catastrophic precision loss
   if the condition number `np.linalg.cond(A)` is large.
   The correct, numerically stable method is `np.linalg.solve(A, b)`, which
   uses LU decomposition with partial pivoting in O(2/3 N^3) time.

3. The Einstein Summation Convention (`np.einsum`):
   Einsum provides a unified, concise, index-based syntax for virtually any
   tensor contraction, trace, transpose, or product:
   Rule 1: Indices repeated across input operands that do NOT appear in the
           output string are summed over (contracted).
   Rule 2: Indices appearing in the output string are retained.

Contents:
---------
1. Matrix Multiplication: Element-wise vs. Matmul vs. Dot vs. Batched Matmul
2. Solving Linear Systems ($Ax = b$), Condition Numbers & Ordinary Least Squares
3. Matrix Decompositions: Cholesky ($LL^T$), QR ($QR$), SVD ($U \\Sigma V^T$) & Eigendecomposition
4. Einstein Summation Convention (`np.einsum`): The Universal Tensor Tool
5. Real-World Case Study: Complete Principal Component Analysis (PCA) via SVD
================================================================================
"""

import time
import numpy as np


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. MATRIX MULTIPLICATION: HADAMARD VS. MATMUL VS. BATCHED TENSORS
# ==============================================================================
print_section("1. Matrix Multiplication: Hadamard vs. Matmul vs. Batched")

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

# Element-wise (Hadamard) product
elem_product = A * B
# Matrix multiplication (@ or np.matmul)
mat_product = A @ B

print(f"Matrix A (2x2):\n{A}")
print(f"Matrix B (2x2):\n{B}")
print(f"\nElement-wise A * B:\n{elem_product}")
print(f"Matrix Multiplication A @ B:\n{mat_product}")

# Batched Matrix Multiplication (Common in Deep Learning Attention Layers):
# Batch size B=2, sequences N=3, feature embedding K=4, output M=5
batch_A = np.random.default_rng(42).standard_normal((2, 3, 4))
batch_B = np.random.default_rng(42).standard_normal((2, 4, 5))

# (2, 3, 4) @ (2, 4, 5) -> (2, 3, 5)
batch_result = batch_A @ batch_B
print(f"\nBatched Tensor Multiplication:")
print(f"Batch A shape: {batch_A.shape}")
print(f"Batch B shape: {batch_B.shape}")
print(f"Batched Output shape: {batch_result.shape}  <-- Batch dimension preserved!")


# ==============================================================================
# 2. SOLVING LINEAR SYSTEMS, CONDITION NUMBERS & LEAST SQUARES
# ==============================================================================
print_section("2. Solving Linear Systems (Ax = b) & Numerical Stability")

# System of Equations:
#  3x +  1y - 2z =  9
#  2x +  8y + 1z = 54
#  1x +  1y + 5z = 11

coeff_matrix = np.array([
    [3.0, 1.0, -2.0],
    [2.0, 8.0,  1.0],
    [1.0, 1.0,  5.0]
])
ordinates = np.array([9.0, 54.0, 11.0])

# ANTI-PATTERN: Explicit Matrix Inversion
x_inv = np.linalg.inv(coeff_matrix) @ ordinates

# BEST PRACTICE: Direct LU Decomposition Solver
x_solve = np.linalg.solve(coeff_matrix, ordinates)

print(f"Coefficients A:\n{coeff_matrix}")
print(f"Constants b: {ordinates}")
print(f"\nSolution via np.linalg.solve (Best Practice): x={x_solve[0]:.2f}, y={x_solve[1]:.2f}, z={x_solve[2]:.2f}")
print(f"Solutions match within machine epsilon: {np.allclose(x_inv, x_solve)}")

# Matrix Condition Number (Detecting Ill-Conditioned Systems)
# A high condition number (>> 1000) warns that minor numerical errors will explode
cond_num = np.linalg.cond(coeff_matrix)
print(f"Matrix Condition Number: {cond_num:.2f} (Well-conditioned: stable inversion)")

# Ordinary Least Squares (Solving Overdetermined Systems: more equations than unknowns)
# Fitting y = m*x + c to noisy experimental observations
x_pts = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_pts = np.array([1.1, 2.9, 5.2, 6.8, 9.1, 11.2])

# Design matrix with bias column of ones: [x, 1]
X_design = np.column_stack([x_pts, np.ones_like(x_pts)])

# Solve least squares: returns (solution, residuals, rank, singular_values)
solution, residuals, rank, s_vals = np.linalg.lstsq(X_design, y_pts, rcond=None)
slope, intercept = solution
print(f"\n--- Overdetermined Linear Regression via np.linalg.lstsq ---")
print(f"Fitted Line: y = {slope:.3f}x + {intercept:.3f}")
print(f"Sum of Squared Residuals: {residuals[0]:.4f}")


# ==============================================================================
# 3. ESSENTIAL MATRIX DECOMPOSITIONS
# ==============================================================================
print_section("3. Essential Matrix Decompositions (Cholesky, QR, SVD, Eig)")

# 1. Cholesky Decomposition: A = L @ L.T
# Requires A to be symmetric and positive-definite (like a Covariance Matrix)
cov_matrix = np.array([
    [4.0, 2.0, 1.0],
    [2.0, 5.0, 3.0],
    [1.0, 3.0, 6.0]
])
L = np.linalg.cholesky(cov_matrix)
print(f"Positive-Definite Covariance Matrix A:\n{cov_matrix}")
print(f"Lower Triangular Cholesky Factor L:\n{np.round(L, 3)}")
print(f"Reconstruction Check (L @ L.T == A): {np.allclose(L @ L.T, cov_matrix)}")

# 2. QR Decomposition: A = Q @ R
# Q is orthogonal (Q.T @ Q = I), R is upper-triangular
Q, R = np.linalg.qr(cov_matrix)
print(f"\nQR Decomposition - Orthogonal Q Check (Q.T @ Q == I):")
print(np.allclose(Q.T @ Q, np.eye(3)))
print(f"Reconstruction Check (Q @ R == A): {np.allclose(Q @ R, cov_matrix)}")

# 3. Singular Value Decomposition (SVD): A = U @ Sigma @ Vh
# Factorizes ANY real rectangular matrix (M x N) into:
# U: (M x M) orthogonal, Sigma: (min(M,N),) singular values, Vh: (N x N) orthogonal
rect_matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
U, S, Vh = np.linalg.svd(rect_matrix)
# Construct full Sigma matrix for reconstruction
Sigma = np.zeros(rect_matrix.shape)
np.fill_diagonal(Sigma, S)
reconstructed_svd = U @ Sigma @ Vh

print(f"\nSVD of Rectangular Matrix {rect_matrix.shape}:")
print(f"Singular Values: {np.round(S, 3)}")
print(f"SVD Reconstruction Check: {np.allclose(rect_matrix, reconstructed_svd)}")

# 4. Eigendecomposition: A @ v = lambda * v
# For symmetric matrices, np.linalg.eigh is faster and guarantees real eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
print(f"\nEigendecomposition (np.linalg.eigh):")
print(f"Eigenvalues:  {np.round(eigenvalues, 3)}")
print(f"First Eigenvector v0: {np.round(eigenvectors[:, 0], 3)}")
# Verification: A @ v0 == lambda0 * v0
v0 = eigenvectors[:, 0]
lambda0 = eigenvalues[0]
print(f"Eigenvector equation verification: {np.allclose(cov_matrix @ v0, lambda0 * v0)}")


# ==============================================================================
# 4. EINSTEIN SUMMATION CONVENTION (`np.einsum`)
# ==============================================================================
print_section("4. Einstein Summation Convention (np.einsum): The Universal Tool")

M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])
v1 = np.array([2, 3])
v2 = np.array([4, 5])

# Standard operations mapped to einsum equivalents:
# 1. Transpose: 'ij->ji'
transposed = np.einsum('ij->ji', M1)
print(f"1. Transpose via einsum('ij->ji'):\n{transposed}")

# 2. Matrix Trace: 'ii->' (Repeated index without output sums the diagonal)
trace_val = np.einsum('ii->', M1)
print(f"2. Trace via einsum('ii->'): {trace_val} (Check np.trace: {np.trace(M1)})")

# 3. Vector Dot Product: 'i,i->'
dot_val = np.einsum('i,i->', v1, v2)
print(f"3. Dot product via einsum('i,i->'): {dot_val} (Check np.dot: {np.dot(v1, v2)})")

# 4. Matrix Multiplication: 'ij,jk->ik'
# 'j' is repeated in input operands and omitted in output -> summed over!
matmul_einsum = np.einsum('ij,jk->ik', M1, M2)
print(f"4. Matmul via einsum('ij,jk->ik'):\n{matmul_einsum}")
print(f"Matches standard @ operator: {np.array_equal(matmul_einsum, M1 @ M2)}")

# 5. Batched Matrix Multiplication: 'bij,bjk->bik'
batch_einsum = np.einsum('bij,bjk->bik', batch_A, batch_B, optimize=True)
print(f"5. Batched matmul via einsum('bij,bjk->bik'): shape {batch_einsum.shape}")
print(f"Matches batch_A @ batch_B: {np.allclose(batch_einsum, batch_result)}")

# 6. Bilinear form: x.T @ A @ y -> 'i,ij,j->'
bilinear = np.einsum('i,ij,j->', v1, M1, v2)
print(f"6. Bilinear form x.T @ A @ y via einsum: {bilinear}")


# ==============================================================================
# 5. REAL-WORLD CASE STUDY: PCA FROM SCRATCH VIA SVD
# ==============================================================================
print_section("5. Case Study: Principal Component Analysis (PCA) via SVD")

# Generate synthetic high-dimensional data: 500 samples, 5 features with correlations
rng = np.random.default_rng(2026)
raw_features = rng.standard_normal((500, 5))
# Inject strong correlation into features
raw_features[:, 1] += 2.0 * raw_features[:, 0]
raw_features[:, 2] -= 1.5 * raw_features[:, 0]

class PrincipalComponentAnalysis:
    """
    Production-grade Principal Component Analysis (PCA) using SVD.
    Avoids explicit covariance matrix calculation for superior numerical stability.
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        # 1. Center the data (mean = 0)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Perform SVD on centered data matrix
        # In SVD: X = U @ Sigma @ Vh. The rows of Vh (columns of V) are the eigenvectors!
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

        # 3. Extract the top n_components
        self.components_ = Vh[:self.n_components]

        # 4. Compute explained variance ratios
        # Variance of each singular value is S^2 / (N - 1)
        variances = (S ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(variances)
        self.explained_variance_ratio_ = (variances / total_variance)[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project centered data onto principal component eigenvectors."""
        assert self.mean_ is not None and self.components_ is not None
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

# Execute PCA
pca = PrincipalComponentAnalysis(n_components=2)
pca.fit(raw_features)
projected_data = pca.transform(raw_features)

print(f"Original Feature Matrix Shape:  {raw_features.shape}")
print(f"Projected 2D Subspace Shape:    {projected_data.shape}")
print(f"Explained Variance Ratio (PC1): {pca.explained_variance_ratio_[0] * 100:.2f}%")
print(f"Explained Variance Ratio (PC2): {pca.explained_variance_ratio_[1] * 100:.2f}%")
print(f"Total Variance Captured (2D):   {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

print("\n[SUCCESS] Part 3 completed successfully.")
