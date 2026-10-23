"""
================================================================================
Day 26: NumPy Deep Dive - Part 2: Broadcasting, Vectorization & ufuncs
================================================================================
Vectorization and Broadcasting form the computational engine of NumPy.
Rather than executing slow, interpreted Python bytecode loops that incur
pointer lookups and dynamic type-checking on every cycle, NumPy executes
tight C/Fortran SIMD (Single Instruction, Multiple Data) loops that run directly
on CPU vector registers.

Theoretical Deep-Dive:
----------------------
1. The Formal Broadcasting Rules:
   NumPy compares the shapes of two arrays element-by-element starting from the
   TRAILING (rightmost) dimension and working backwards:
   Two dimensions are compatible if:
     a. They are equal, OR
     b. One of them is 1.
   If one array has fewer dimensions, its shape is prepended with 1s until
   the dimension counts match.

2. Zero-Stride Virtual Memory Stretching:
   When an array dimension of size 1 is broadcast to size N, NumPy does NOT
   copy the data N times in memory. Instead, it creates a new view where the
   stride for that dimension is set to 0 bytes!
   Address = Base + (Index * 0) = Base.
   The CPU continuously re-reads the same memory cache line, resulting in
   instantaneous, zero-memory expansion.

3. Universal Functions (ufuncs):
   A ufunc is a C-implemented wrapper around a function that operates
   element-by-element on `ndarray` objects. ufuncs support:
   - Type coercion & casting rules
   - Dimension broadcasting
   - High-performance reduction methods (`.reduce()`, `.accumulate()`, `.outer()`, `.at()`)
   - Direct memory destination routing via the `out=` argument to avoid RAM churn.

Contents:
---------
1. The Formal Broadcasting Rules & Zero-Stride Memory Analysis
2. Core Universal Functions (ufuncs) & the Memory-Saving `out=` Parameter
3. Advanced ufunc Methods: reduce, accumulate, outer, reduceat, and unbuffered `.at()`
4. Native Vectorization vs. The `np.vectorize` Wrapper Myth
5. Real-World Applications: Pairwise Euclidean Distance Matrix (Zero Python Loops)
6. Floating-Point Subtleties: IEEE 754, NaNs, Infs, Machine Epsilon & Error Handling
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
# 1. THE FORMAL BROADCASTING RULES & ZERO-STRIDE MEMORY ANALYSIS
# ==============================================================================
print_section("1. Formal Broadcasting Rules & Zero-Stride Virtual Memory")

# Case A: 2D Matrix (3, 4) + 1D Row Vector (4,)
# Step 1: Prepend 1s -> (3, 4) and (1, 4)
# Step 2: Dimension 1 matches (4 == 4). Dimension 0 has size 1, broadcasted to 3.
matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])
row_vec = np.array([1, 2, 3, 4])
sum_result = matrix + row_vec

print("Case A: (3, 4) + (4,):")
print(f"Matrix shape:  {matrix.shape}")
print(f"Row vector:    {row_vec.shape}")
print(f"Result shape:  {sum_result.shape}")
print(f"Result Matrix:\n{sum_result}")

# Case B: Column Vector (3, 1) + Row Vector (1, 4) -> Outer Grid (3, 4)
col_vec = np.array([[100], [200], [300]])  # Shape (3, 1)
row_vec2 = np.array([[1, 2, 3, 4]])        # Shape (1, 4)
grid_sum = col_vec + row_vec2

print("\nCase B: Outer Addition Grid (3, 1) + (1, 4):")
print(f"Result shape: {grid_sum.shape}")
print(f"Grid Matrix:\n{grid_sum}")

# Zero-Stride Demonstration:
# How does NumPy stretch `col_vec` along columns without allocating memory?
# Stride for column axis is 0 bytes!
broadcast_view = np.broadcast_to(col_vec, (3, 4))
print("\n--- Zero-Stride Inspection ---")
print(f"Original col_vec strides:       {col_vec.strides}  (8 bytes row, 8 bytes col)")
print(f"Broadcasted view strides:      {broadcast_view.strides} (8 bytes row, 0 BYTES col!)")
print(f"Broadcast view shares memory?   {np.shares_memory(col_vec, broadcast_view)} (True)")
print(f"col_vec nbytes: {col_vec.nbytes} bytes vs. broadcast_view virtual nbytes: {broadcast_view.nbytes} bytes")


# ==============================================================================
# 2. CORE UFUNCS & THE MEMORY-SAVING `out=` PARAMETER
# ==============================================================================
print_section("2. Core Universal Functions (ufuncs) & Memory-Saving 'out='")

x = np.linspace(0.0, 2.0 * np.pi, 5)
print(f"Source angles (x):       {np.round(x, 3)}")

# Unary ufuncs
sin_x = np.sin(x)
exp_x = np.exp(x[:3])
log_x = np.log(x[1:4])
print(f"np.sin(x):               {np.round(sin_x, 3)}")
print(f"np.exp(x[:3]):           {np.round(exp_x, 3)}")
print(f"np.log(x[1:4]):          {np.round(log_x, 3)}")

# Binary ufuncs
a = np.array([10.0, -25.0, 30.0])
b = np.array([12.0, 20.0, -5.0])
print(f"\nnp.maximum(a, b):        {np.maximum(a, b)}")
print(f"np.copysign(a, b):       {np.copysign(a, b)}  (copy sign of b to magnitude of a)")

# The `out=` Memory Buffer Optimization:
# When executing multiple array operations, standard Python generates temporary
# intermediate arrays that thrash the garbage collector.
# Using `out=` reuses pre-allocated memory buffers directly!
large_a = np.ones(1_000_000, dtype=np.float64)
large_b = np.full(1_000_000, 2.0, dtype=np.float64)
result_buffer = np.empty_like(large_a)

# Standard: creates temporary array in heap
t0 = time.perf_counter()
for _ in range(20):
    temp = large_a * 5.0 + large_b
t_standard = time.perf_counter() - t0

# Optimized: writes directly into pre-allocated buffer
t0 = time.perf_counter()
for _ in range(20):
    np.multiply(large_a, 5.0, out=result_buffer)
    np.add(result_buffer, large_b, out=result_buffer)
t_optimized = time.perf_counter() - t0

print(f"\nBenchmark (20 iterations across 1,000,000 floats):")
print(f"Standard chained expression:  {t_standard * 1000:.2f} ms")
print(f"In-place 'out=' reuse:        {t_optimized * 1000:.2f} ms")
print(f"Speedup from zero-allocation: {t_standard / t_optimized:.2f}x")


# ==============================================================================
# 3. ADVANCED UFUNC METHODS: REDUCE, ACCUMULATE, OUTER, REDUCEAT, AT
# ==============================================================================
print_section("3. Advanced ufunc Methods: reduce, accumulate, outer, reduceat, at")

arr = np.array([1, 2, 3, 4, 5])
matrix_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 1. reduce: collapse array along an axis using the ufunc operator
sum_val = np.add.reduce(arr)             # Same as arr.sum()
product_val = np.multiply.reduce(arr)   # Same as arr.prod()
col_sums = np.add.reduce(matrix_2d, axis=0)
print(f"np.add.reduce([1, 2, 3, 4, 5]):      {sum_val}")
print(f"np.multiply.reduce([1, 2, 3, 4, 5]): {product_val}")
print(f"Column-wise sum via reduce:          {col_sums}")

# 2. accumulate: running / prefix cumulative calculation
cum_sum = np.add.accumulate(arr)
cum_prod = np.multiply.accumulate(arr)
print(f"\nnp.add.accumulate:      {cum_sum}")
print(f"np.multiply.accumulate: {cum_prod}")

# 3. outer: compute the ufunc for all pairs of elements
multiplication_table = np.multiply.outer(np.arange(1, 6), np.arange(1, 6))
print(f"\nOuter Multiplication Table (5x5):\n{multiplication_table}")

# 4. reduceat: performs localized reductions over slice index intervals
# Indices [0, 2, 4] means slices: arr[0:2], arr[2:4], arr[4:]
segmented_sums = np.add.reduceat(arr, [0, 2, 4])
print(f"\nnp.add.reduceat on arr with indices [0, 2, 4]: {segmented_sums}")
print(f"Verification: [1+2, 3+4, 5] = [3, 7, 5]")

# 5. .at(): unbuffered in-place updates (handles repeated indices correctly!)
# Standard Python/NumPy assignment `arr[[0, 0]] += 1` increments index 0 only ONCE!
# ufunc.at() correctly increments index 0 TWICE!
counter = np.zeros(5, dtype=np.int32)
duplicate_indices = np.array([1, 2, 1, 4, 1, 2])

# Standard syntax fails with duplicate indices:
flawed = np.zeros(5, dtype=np.int32)
flawed[duplicate_indices] += 1
print(f"\nStandard indexing with duplicates (FLAWED): {flawed} (Notice index 1 was updated only once!)")

# Correct: np.add.at performs unbuffered accumulation
np.add.at(counter, duplicate_indices, 1)
print(f"np.add.at with duplicates (CORRECT):         {counter} (Index 1 count is 3, Index 2 count is 2)")


# ==============================================================================
# 4. NATIVE VECTORIZATION VS. THE `np.vectorize` WRAPPER MYTH
# ==============================================================================
print_section("4. Native Vectorization vs. The 'np.vectorize' Wrapper Myth")

# Important Architectural Fact:
# `np.vectorize` is NOT a JIT compiler. It does NOT convert Python code to C.
# It is merely a convenience wrapper around a Python `for` loop!

def custom_scalar_activation(x: float) -> float:
    """Non-linear piecewise activation function."""
    if x > 0:
        return x ** 1.5
    else:
        return 0.1 * x

test_data = np.random.default_rng(42).standard_normal(100_000)

# Approach A: np.vectorize wrapper (Still pure Python underneath!)
vectorized_wrapper = np.vectorize(custom_scalar_activation)
t0 = time.perf_counter()
res_wrapper = vectorized_wrapper(test_data)
t_wrapper = time.perf_counter() - t0

# Approach B: Native NumPy vectorized formulation (True C-speed SIMD)
# GOTCHA NOTE: In `np.where(cond, a, b)`, both expressions `a` and `b` are fully evaluated!
# Evaluating `test_data ** 1.5` on negative numbers generates invalid float warnings.
# The correct idiomatic solution is clipping before power: `np.maximum(test_data, 0.0) ** 1.5`.
t0 = time.perf_counter()
pos_part = np.maximum(test_data, 0.0) ** 1.5
neg_part = 0.1 * np.minimum(test_data, 0.0)
res_native = pos_part + neg_part
t_native = time.perf_counter() - t0

print(f"Data elements: {len(test_data):,}")
print(f"np.vectorize wrapper execution time: {t_wrapper * 1000:.2f} ms")
print(f"Native NumPy vectorized time:        {t_native * 1000:.2f} ms")
print(f"Real Speedup factor:                 {t_wrapper / t_native:.1f}x FASTER!")
print(f"Results match within tolerance:      {np.allclose(res_wrapper, res_native)}")


# ==============================================================================
# 5. REAL-WORLD APPLICATION: PAIRWISE EUCLIDEAN DISTANCE MATRIX
# ==============================================================================
print_section("5. Real-World Application: Pairwise Euclidean Distance Matrix")

# Problem: Given N points in D-dimensional space, compute the pairwise distance
# matrix of shape (N, N) where dist[i, j] = ||P_i - P_j||_2.
# Naive approach uses O(N^2) nested Python loops.
# NumPy solves this in a single broadcasted vectorized expression!

rng = np.random.default_rng(2026)
points = rng.uniform(0.0, 100.0, size=(1000, 3))  # 1000 3D spatial coordinates

# Broadcasting trick:
# points[:, np.newaxis, :] has shape (1000, 1, 3)
# points[np.newaxis, :, :] has shape (1, 1000, 3)
# Difference has shape (1000, 1000, 3)
# Sum of squares across axis -1 has shape (1000, 1000)

t0 = time.perf_counter()
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
t_dist = time.perf_counter() - t0

print(f"Points: {points.shape[0]} in {points.shape[1]}D space")
print(f"Computed 1,000,000 pairwise distances in: {t_dist * 1000:.2f} ms")
print(f"Distance matrix shape: {dist_matrix.shape}")
print(f"Diagonal check (dist[i, i] must be 0): {np.allclose(np.diag(dist_matrix), 0.0)}")
print(f"Symmetry check (dist[i, j] == dist[j, i]): {np.allclose(dist_matrix, dist_matrix.T)}")
print(f"Average pairwise distance: {np.mean(dist_matrix):.2f} units")


# ==============================================================================
# 6. FLOATING-POINT SUBTLETIES: IEEE 754, NANS, INFS & ERROR CONTROL
# ==============================================================================
print_section("6. Floating-Point Subtleties, IEEE 754 & Error Control")

# 1. Why direct equality `==` fails with floating-point values
val_a = 0.1 + 0.2
val_b = 0.3
print(f"0.1 + 0.2 == 0.3 in standard float: {val_a == val_b} (False due to binary float representation!)")
print(f"np.isclose(0.1 + 0.2, 0.3):         {np.isclose(val_a, val_b)} (True via relative/absolute tolerance)")

# 2. IEEE 754 Special Values: NaN and Inf
values = np.array([1.0, 0.0, -1.0])
divisors = np.array([0.0, 0.0, 0.0])

# Using np.errstate to handle hardware floating-point exceptions gracefully
with np.errstate(divide='ignore', invalid='ignore'):
    ieee_results = values / divisors

print(f"\nIEEE 754 results of [1/0, 0/0, -1/0]:\n{ieee_results}")
print(f"np.isnan flags:   {np.isnan(ieee_results)}")
print(f"np.isinf flags:   {np.isinf(ieee_results)}")
print(f"np.isfinite flags:{np.isfinite(ieee_results)}")

# NaN comparison rule: np.nan == np.nan is ALWAYS False!
nan_val = np.nan
print(f"\nnp.nan == np.nan: {nan_val == nan_val} (Always False by IEEE 754 specification!)")
print(f"Correct check:    np.isnan(nan_val) -> {np.isnan(nan_val)}")

# Nan-aware reduction functions
data_with_nans = np.array([10.0, np.nan, 20.0, 30.0, np.nan])
print(f"\nStandard np.mean (poisoned by NaN): {np.mean(data_with_nans)}")
print(f"Nan-aware np.nanmean:               {np.nanmean(data_with_nans)}")
print(f"Nan-aware np.nanstd:                {np.nanstd(data_with_nans):.2f}")

print("\n[SUCCESS] Part 2 completed successfully.")
