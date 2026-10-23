"""
================================================================================
Day 26: NumPy Deep Dive - Part 4: Randomness, Statistics, Sorting & Binary I/O
================================================================================
From generating stochastic Monte Carlo paths and computing multi-dimensional
statistical metrics, to O(N) partitioning and memory-mapping gigabyte-scale
datasets directly from disk, NumPy provides an industrial-grade toolset for
computational data workflows.

Theoretical Deep-Dive:
----------------------
1. The Modern Random API (default_rng vs Legacy np.random):
   Prior to NumPy 1.17, randomness relied on the global Mersenne Twister (MT19937)
   algorithm via `np.random.seed()`. MT19937 had severe limitations:
   - Global mutable state (non-thread-safe race conditions).
   - Poor statistical properties in high dimensions (failed BigCrush battery).
   - Slow stream generation.
   NumPy's modern API uses `np.random.default_rng()`, powered by the PCG64
   (Permuted Congruential Generator) bit generator:
   - Thread-safe, isolated state instances.
   - Passes all modern statistical randomness suites.
   - Up to 4x faster random float/integer generation.

2. O(N) Quickselect vs O(N log N) Full Sorting:
   To find the top-K elements in an array of size N, developers frequently
   sort the entire array with `np.sort()` in O(N log N) time.
   NumPy provides `np.partition()` and `np.argpartition()`, implementing the
   C-level Introselect algorithm. It guarantees that the K-th element is placed
   at its sorted position, with smaller elements to its left and larger to its
   right, completing in strictly linear O(N) time!

3. Zero-Copy Out-of-Core Processing with `np.memmap`:
   When arrays exceed available physical RAM, `np.memmap` uses the operating
   system's virtual memory subsystem (mmap on POSIX, CreateFileMapping on Windows)
   to treat a binary file on disk as an array buffer. Memory pages are read
   and written lazily on demand without loading the full file.

Contents:
---------
1. Modern Random Generator API: PCG64, Continuous & Discrete Distributions, Sampling
2. Descriptive Statistics, Degrees of Freedom, Covariance & Quantiles
3. O(N) Partitioning, Argpartition, Lexsort, and Binary Search (searchsorted)
4. Serialization (.npy, .npz) & Out-of-Core Processing with `np.memmap`
5. Real-World Case Study: Real-Time Sliding Telemetry Anomaly Detector
================================================================================
"""

import os
import time
import numpy as np


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. MODERN RANDOM GENERATOR API: PCG64 & DISTRIBUTIONS
# ==============================================================================
print_section("1. Modern Random Generator API (default_rng vs Legacy)")

# Initialize reproducible Generator instance
rng = np.random.default_rng(seed=2026)

# Continuous Distributions
standard_normals = rng.standard_normal(size=(3, 3))
uniform_floats = rng.uniform(low=10.0, high=50.0, size=5)
beta_distribution = rng.beta(a=2.0, b=5.0, size=5)
exponential_times = rng.exponential(scale=1.5, size=5)

print(f"Standard Normal Matrix (3x3):\n{np.round(standard_normals, 2)}")
print(f"\nUniform Floats [10.0, 50.0): {np.round(uniform_floats, 2)}")
print(f"Beta Distribution (a=2, b=5): {np.round(beta_distribution, 3)}")
print(f"Exponential Inter-arrival:   {np.round(exponential_times, 3)}")

# Discrete Distributions
integers = rng.integers(low=1, high=100, size=6)
poisson_counts = rng.poisson(lam=4.5, size=6)  # Arrival counts (rate lambda=4.5)
binomial_trials = rng.binomial(n=10, p=0.5, size=6)  # Coin flips

print(f"\nDiscrete Integers [1, 100):  {integers}")
print(f"Poisson Process (lambda=4.5):{poisson_counts}")
print(f"Binomial Trials (n=10, p=.5):{binomial_trials}")

# Weighted Sampling & Permutation
assets = np.array(["NVDA", "AAPL", "MSFT", "GOOGL"])
probabilities = np.array([0.40, 0.30, 0.20, 0.10])  # Must sum to 1.0

# Sampling with replacement
sampled_portfolio = rng.choice(assets, size=8, replace=True, p=probabilities)
print(f"\nWeighted Portfolio Sample (8 picks): {sampled_portfolio}")

# In-place shuffle vs immutable permutation
deck = np.arange(10)
shuffled_copy = rng.permutation(deck)  # Returns new permuted array
rng.shuffle(deck)                      # Mutates deck in place
print(f"Permuted copy:  {shuffled_copy}")
print(f"Shuffled array: {deck}")


# ==============================================================================
# 2. DESCRIPTIVE STATISTICS, DEGREES OF FREEDOM & CORRELATION
# ==============================================================================
print_section("2. Descriptive Statistics, Degrees of Freedom & Correlation")

# Generate 100,000 observations
data = rng.normal(loc=100.0, scale=15.0, size=100_000)

mean_val = np.mean(data)
median_val = np.median(data)

# Population Variance vs Sample Variance (Degrees of Freedom)
# ddof=0: Divides by N (Biased / Population Variance)
# ddof=1: Divides by N - 1 (Bessel's Correction for Sample Variance)
pop_var = np.var(data, ddof=0)
sample_var = np.var(data, ddof=1)
sample_std = np.std(data, ddof=1)

print(f"Sample Count: {len(data):,}")
print(f"Mean:         {mean_val:.3f}")
print(f"Median:       {median_val:.3f}")
print(f"Population Variance (ddof=0): {pop_var:.3f}")
print(f"Sample Variance (ddof=1):     {sample_var:.3f}")
print(f"Sample Std Dev (ddof=1):      {sample_std:.3f}")

# Quantiles and Percentiles
q25, q50, q75 = np.percentile(data, [25, 50, 75])
iqr = q75 - q25
print(f"\n25th Percentile (Q1): {q25:.2f}")
print(f"50th Percentile (Q2): {q50:.2f}")
print(f"75th Percentile (Q3): {q75:.2f}")
print(f"Interquartile Range:  {iqr:.2f}")

# Covariance & Correlation Matrix
x_series = rng.standard_normal(1000)
y_series = 2.5 * x_series + rng.normal(0, 0.5, 1000)  # Strong positive correlation

cov_matrix = np.cov(x_series, y_series)
corr_matrix = np.corrcoef(x_series, y_series)
print(f"\nCovariance Matrix (2x2):\n{np.round(cov_matrix, 3)}")
print(f"Pearson Correlation Coefficient: {corr_matrix[0, 1]:.4f} (Near 1.0 -> High positive correlation)")


# ==============================================================================
# 3. HIGH-PERFORMANCE SORTING, O(N) PARTITIONING & BINARY SEARCH
# ==============================================================================
print_section("3. O(N) Partitioning, Argpartition & Binary Search")

# Benchmark: Top-K items via Full Sort O(N log N) vs. Argpartition O(N)
N = 1_000_000
K = 5
large_scores = rng.uniform(0.0, 1000.0, size=N)

# Method A: Full Sort (Anti-Pattern for Top-K)
t0 = time.perf_counter()
sorted_indices = np.argsort(large_scores)
top_k_sort = sorted_indices[-K:][::-1]
t_sort = time.perf_counter() - t0

# Method B: Linear Time Argpartition (Best Practice)
t0 = time.perf_counter()
# Places the (N - K)-th element in its exact position; all elements to its right are >=
part_indices = np.argpartition(large_scores, -K)[-K:]
# Sort only the small K elements
top_k_part = part_indices[np.argsort(-large_scores[part_indices])]
t_part = time.perf_counter() - t0

print(f"Array size: {N:,} elements. Finding Top-{K} elements:")
print(f"Method A (np.argsort full sort):     {t_sort * 1000:.2f} ms")
print(f"Method B (np.argpartition quickselect): {t_part * 1000:.2f} ms")
print(f"Speedup Factor:                      {t_sort / t_part:.1f}x FASTER!")
print(f"Identical Top-{K} values confirmed:  {np.array_equal(top_k_sort, top_k_part)}")

# Binary Search in Sorted Arrays via np.searchsorted
# Finds the insertion indices to maintain sorted order in O(log N)
sorted_thresholds = np.array([10.0, 25.0, 50.0, 75.0, 100.0])
incoming_readings = np.array([5.0, 25.0, 42.0, 80.0, 105.0])
bucket_indices = np.searchsorted(sorted_thresholds, incoming_readings, side='right')
print(f"\n--- O(log N) Binary Search Bucket Assignment ---")
print(f"Thresholds: {sorted_thresholds}")
print(f"Readings:   {incoming_readings}")
print(f"Buckets:    {bucket_indices}")

# Multi-Key Indirect Sort with np.lexsort
# Primary sort: department (col 0), Secondary tie-breaker: salary (col 1)
depts = np.array(['IT', 'HR', 'IT', 'Sales', 'HR', 'IT'])
salaries = np.array([85000, 62000, 95000, 71000, 68000, 85000])

# Note: lexsort expects sort keys in REVERSE priority order: (secondary, primary)
sort_order = np.lexsort((salaries, depts))
print(f"\nLexsort order by Dept then Salary:")
for idx in sort_order:
    print(f"  {depts[idx]:<6} | ${salaries[idx]:,}")


# ==============================================================================
# 4. BINARY SERIALIZATION (.NPY/.NPZ) & OUT-OF-CORE MEMMAP
# ==============================================================================
print_section("4. Binary Serialization & Out-of-Core Processing (np.memmap)")

temp_npy = "temp_dataset.npy"
temp_npz = "temp_archive.npz"
temp_dat = "temp_mmap.dat"

try:
    # 1. Single Array Fast Binary I/O (.npy)
    # Writes pure contiguous memory bytes with an exact dtype/shape header
    raw_matrix = rng.normal(0, 1, size=(500, 100))
    np.save(temp_npy, raw_matrix)
    loaded_matrix = np.load(temp_npy)
    print(f".npy Save/Load integrity verified: {np.array_equal(raw_matrix, loaded_matrix)}")

    # 2. Compressed Multi-Array Archive (.npz)
    weights = rng.standard_normal((10, 5))
    biases = np.zeros(5)
    np.savez_compressed(temp_npz, weights=weights, biases=biases)

    with np.load(temp_npz) as archive:
        print(f".npz Keys: {archive.files}")
        print(f"Recovered weights shape: {archive['weights'].shape}, biases: {archive['biases'].shape}")

    # 3. Out-of-Core Processing with np.memmap
    # Creates an array on disk that behaves identically to RAM
    # Allows processing terabyte-scale datasets without memory exhaustion!
    shape = (1000, 1000)
    dtype = np.float64

    # Allocate disk-backed array
    mmap_arr = np.memmap(temp_dat, dtype=dtype, mode='w+', shape=shape)
    mmap_arr[:] = 0.0  # Zero initialization on disk

    # Write localized patch
    mmap_arr[10:15, 10:15] = 999.0
    mmap_arr.flush()  # Force OS to write dirty memory pages to disk
    del mmap_arr      # Close file mapping

    # Re-open in read-only mode without loading file to RAM
    mmap_read = np.memmap(temp_dat, dtype=dtype, mode='r', shape=shape)
    patch_mean = np.mean(mmap_read[10:15, 10:15])
    print(f"\nnp.memmap read-back patch mean: {patch_mean} (Expected 999.0)")
    del mmap_read
    import gc
    gc.collect()  # Release file mapping handles on Windows

finally:
    # Cleanup temporary test files
    for f in [temp_npy, temp_npz, temp_dat]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    print("Temporary binary files cleaned up.")


# ==============================================================================
# 5. REAL-WORLD CASE STUDY: SLIDING TELEMETRY ANOMALY DETECTOR
# ==============================================================================
print_section("5. Case Study: Telemetry Anomaly Detector (Sliding Percentiles)")

# Scenario: An industrial turbine streams vibration sensor data.
# We must flag anomalous spikes exceeding the 99th percentile without lag.

# Stream of 5,000 sensor readings
stream = rng.normal(loc=50.0, scale=2.0, size=5000)
# Inject 10 anomalous vibration shocks at random intervals
anomaly_indices = rng.choice(5000, size=10, replace=False)
stream[anomaly_indices] += rng.uniform(15.0, 30.0, size=10)

def detect_outliers_vectorized(readings: np.ndarray, threshold_std: float = 3.5) -> dict:
    """Detects statistical outliers using median and Median Absolute Deviation (MAD)."""
    med = np.median(readings)
    # MAD is a robust measure of statistical dispersion
    mad = np.median(np.abs(readings - med))
    # Modified Z-score: 0.6745 is the consistency factor for normal distributions
    modified_z_scores = 0.6745 * (readings - med) / (mad + 1e-9)

    outlier_mask = np.abs(modified_z_scores) > threshold_std
    outlier_indices = np.nonzero(outlier_mask)[0]

    return {
        "median": med,
        "mad": mad,
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices,
        "outlier_values": readings[outlier_indices]
    }

detection = detect_outliers_vectorized(stream)
print(f"Turbine Sensor Stream: {len(stream):,} data points")
print(f"Baseline Median: {detection['median']:.2f}, Robust MAD: {detection['mad']:.2f}")
print(f"Anomalies Injected: 10 | Anomalies Flagged: {detection['outlier_count']}")
print(f"First 5 Flagged Values: {np.round(detection['outlier_values'][:5], 2)}")

print("\n[SUCCESS] Part 4 completed successfully.")
