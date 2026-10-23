"""
================================================================================
Day 26: NumPy Deep Dive - Part 1: Arrays, Memory Architecture, Strides & Indexing
================================================================================
NumPy (Numerical Python) is the foundational linchpin of scientific computing,
data science, and machine learning in Python. At its core is the `ndarray`
(N-dimensional array), a data structure engineered for raw performance, SIMD
hardware acceleration, and zero-overhead memory manipulation.

Theoretical Deep-Dive:
----------------------
1. The Anatomy of an ndarray:
   An `ndarray` object consists of two core components:
   a. Data Buffer: A contiguous block of raw memory bytes allocated via malloc().
   b. Metadata Descriptor (The Array Header):
      - `data`: Pointer to the first byte in memory.
      - `dtype`: Description of the data type and byte size per element.
      - `shape`: A tuple indicating the length along each dimension (axis).
      - `strides`: A tuple indicating the number of bytes to step in memory to
                  advance by one element along each axis.
      - `flags`: Memory layout flags (C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA, WRITEABLE).
      - `base`: Pointer to the parent array if this array is a memory view.

2. Striding Mechanics:
   In a 2D array of shape (3, 4) with 64-bit integers (8 bytes per item):
   - C-order (Row-Major): Row stride = 4 * 8 = 32 bytes; Column stride = 8 bytes.
     Strides: (32, 8).
     Memory Address for (i, j) = base_address + (i * 32) + (j * 8).
   - Fortran-order (Column-Major): Column stride = 3 * 8 = 24 bytes; Row stride = 8 bytes.
     Strides: (8, 24).

3. Views vs. Copies:
   Slicing an array (e.g. `a[1:3, :]`) merely creates a new header with modified
   shape, strides, and data pointer pointing into the SAME existing memory buffer.
   No data is copied. In contrast, "Fancy Indexing" (using integer or boolean arrays)
   always allocates brand-new memory and copies the data.

Contents:
---------
1. Memory Architecture: Python Lists vs. C-Contiguous ndarrays (Footprint & Cache)
2. Array Creation Mechanics (empty, zeros, ones, full, arange, linspace, logspace, buffers)
3. Data Types (Dtypes), Precision Overflows, and C-Style Structured Arrays
4. Memory Strides, Views vs. Copies, and Preventing Memory Leak Traps
5. Advanced Indexing: Slicing, Boolean Masking, Fancy Indexing, np.where, np.select
6. Shape Engineering: Reshape, Ravel vs. Flatten, Transpose, Stacking & Splitting
================================================================================
"""

import sys
import time
import numpy as np


def print_section(title: str) -> None:
    """Helper utility for formatted console section banners."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. MEMORY ARCHITECTURE: PYTHON LISTS VS. C-CONTIGUOUS NDARRAY
# ==============================================================================
print_section("1. Memory Architecture: Python Lists vs. NumPy ndarray")

# In standard Python, a list is an array of 8-byte pointers pointing to separate
# PyObject instances scattered throughout heap memory.
# In NumPy, elements are packed sequentially in a contiguous buffer.

NUM_ELEMENTS = 1_000_000

# Benchmark Memory Footprint
py_list = list(range(NUM_ELEMENTS))
# Size of list pointer buffer + sum of size of individual int PyObjects
py_list_bytes = sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:1000]) * (NUM_ELEMENTS // 1000)

np_array_int64 = np.arange(NUM_ELEMENTS, dtype=np.int64)
np_array_int32 = np.arange(NUM_ELEMENTS, dtype=np.int32)

print(f"Elements: {NUM_ELEMENTS:,}")
print(f"Python list estimated memory: {py_list_bytes / (1024 * 1024):.2f} MB")
print(f"NumPy int64 array memory:     {np_array_int64.nbytes / (1024 * 1024):.2f} MB")
print(f"NumPy int32 array memory:     {np_array_int32.nbytes / (1024 * 1024):.2f} MB")
print(f"Memory Savings Factor:        ~{py_list_bytes / np_array_int32.nbytes:.1f}x smaller")

# Inspecting ndarray flags and contiguity
matrix_c = np.ones((4, 5), dtype=np.float64, order='C')
matrix_f = np.ones((4, 5), dtype=np.float64, order='F')

print("\n--- Array Layout & Flags Inspection ---")
print(f"Matrix C-Order Strides (row-major):    {matrix_c.strides}  (bytes to step [row, col])")
print(f"Matrix C-Order C_CONTIGUOUS Flag:      {matrix_c.flags['C_CONTIGUOUS']}")
print(f"Matrix F-Order Strides (column-major): {matrix_f.strides}  (bytes to step [row, col])")
print(f"Matrix F-Order F_CONTIGUOUS Flag:      {matrix_f.flags['F_CONTIGUOUS']}")


# ==============================================================================
# 2. ARRAY CREATION MECHANICS
# ==============================================================================
print_section("2. Array Creation Mechanics & Initialization Strategies")

# 1. Uninitialized memory (Fastest, does not wipe memory - contains garbage values)
raw_buffer = np.empty((2, 3), dtype=np.float64)
print(f"np.empty((2, 3)) [Raw memory garbage]:\n{raw_buffer}")

# 2. Constant initialization
zeros_arr = np.zeros((2, 3), dtype=np.int32)
ones_arr = np.ones((2, 3), dtype=np.float32)
full_arr = np.full((2, 3), fill_value=42.0, dtype=np.float64)
print(f"\nnp.full((2, 3), 42.0):\n{full_arr}")

# 3. Numeric Sequences: arange vs. linspace vs. logspace
# np.arange: step-based (can have floating point rounding issues)
seq_arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
# np.linspace: sample-count based (exact endpoint control)
seq_linspace = np.linspace(0.0, 1.0, num=5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
# np.logspace: logarithmic progression (10^start to 10^stop)
seq_logspace = np.logspace(1, 4, num=4)  # [10, 100, 1000, 10000]

print(f"\nnp.arange(0, 10, 2):      {seq_arange}")
print(f"np.linspace(0, 1, 5):     {seq_linspace}")
print(f"np.logspace(1, 4, 4):     {seq_logspace}")

# 4. Creating arrays from raw byte buffers and Python iterables
byte_stream = b"\x01\x00\x00\x00\x05\x00\x00\x00\x09\x00\x00\x00"
arr_from_bytes = np.frombuffer(byte_stream, dtype=np.int32)
print(f"\nnp.frombuffer from raw byte stream: {arr_from_bytes}")

# 5. Coordinate Grids
x_coords = np.linspace(-2, 2, 5)
y_coords = np.linspace(-1, 1, 3)
xx, yy = np.meshgrid(x_coords, y_coords)
print(f"\nMeshgrid XX shape: {xx.shape}, YY shape: {yy.shape}")
print(f"Radius squared grid (xx^2 + yy^2):\n{np.round(xx**2 + yy**2, 2)}")


# ==============================================================================
# 3. DATA TYPES (DTYPES), OVERFLOW TRAPS & STRUCTURED ARRAYS
# ==============================================================================
print_section("3. Data Types, Overflow Traps & C-Style Structured Arrays")

# Overflow Trap Demonstration:
# Fixed-width integer types wrap around silently without throwing exceptions!
small_int = np.array([254, 255], dtype=np.uint8)
print(f"Initial uint8 array:     {small_int}")
print(f"uint8 after adding 1:    {small_int + np.uint8(1)}  <-- Notice 255 wraps to 0!")
print(f"uint8 after adding 2:    {small_int + np.uint8(2)}  <-- Overflow behavior!")

# Floating point epsilon (machine precision limit)
finfo_f32 = np.finfo(np.float32)
finfo_f64 = np.finfo(np.float64)
print(f"\nFloat32 Epsilon: {finfo_f32.eps:.2e}, Max: {finfo_f32.max:.2e}")
print(f"Float64 Epsilon: {finfo_f64.eps:.2e}, Max: {finfo_f64.max:.2e}")

# Structured Arrays (C-style Structs in Python)
# Useful for high-throughput heterogenous record processing without overhead
sensor_record_dtype = np.dtype([
    ('sensor_id', 'U10'),      # Unicode string up to 10 chars
    ('timestamp', 'datetime64[s]'),
    ('temperature', 'f4'),     # 32-bit float
    ('pressure', 'f8'),        # 64-bit float
    ('active', '?')            # Boolean
])

telemetry_data = np.array([
    ('SENSOR_ALPHA', '2026-10-23T12:00:00', 23.4, 1013.25, True),
    ('SENSOR_BETA',  '2026-10-23T12:00:01', 25.1, 1012.80, False),
    ('SENSOR_GAMMA', '2026-10-23T12:00:02', 22.9, 1014.10, True)
], dtype=sensor_record_dtype)

print("\n--- Structured Array (Sensor Telemetry) ---")
print(telemetry_data)
print(f"Temperatures only: {telemetry_data['temperature']}")
print(f"Active sensors:    {telemetry_data[telemetry_data['active']]['sensor_id']}")


# ==============================================================================
# 4. STRIDES, VIEWS VS. COPIES & PREVENTING MEMORY LEAKS
# ==============================================================================
print_section("4. Strides, Views vs. Copies & Preventing Memory Leaks")

# How Strides Control Dimension Interpretation
base_array = np.arange(12, dtype=np.int32)
reshaped_view = base_array.reshape(3, 4)

print(f"Base Array Memory Strides:     {base_array.strides}  (4 bytes per step)")
print(f"Reshaped View Memory Strides: {reshaped_view.strides} (16 bytes per row, 4 bytes per col)")

# Check if reshaped array shares memory
print(f"reshaped_view.base is base_array: {reshaped_view.base is base_array}")
print(f"np.shares_memory(base, view):    {np.shares_memory(base_array, reshaped_view)}")

# Modifying a VIEW modifies the PARENT array!
reshaped_view[0, 0] = 999
print(f"base_array[0] after modifying reshaped_view[0, 0]: {base_array[0]} (Mutated!)")

# Reset
base_array[0] = 0

# The Critical Memory Leak Gotcha:
# If you extract a small slice from a massive 1GB array and keep a reference to it,
# the ENTIRE 1GB parent remains pinned in RAM because `small_slice.base` holds the parent!
def extract_header_leak_risk(large_arr: np.ndarray) -> np.ndarray:
    return large_arr[:5]  # VIEW! Retains pointer to large_arr in .base

def extract_header_safe(large_arr: np.ndarray) -> np.ndarray:
    return large_arr[:5].copy()  # DEEP COPY! Frees the large parent from RAM

giant_array = np.ones(5_000_000, dtype=np.float64)  # ~40 MB
risky_slice = extract_header_leak_risk(giant_array)
safe_slice = extract_header_safe(giant_array)

print(f"\nRisky slice base is None? {risky_slice.base is None} (False -> Holds giant parent in memory!)")
print(f"Safe slice base is None?  {safe_slice.base is None}  (True -> Independent memory allocation)")


# ==============================================================================
# 5. ADVANCED INDEXING: SLICING, BOOLEAN MASKING, FANCY INDEXING
# ==============================================================================
print_section("5. Advanced Indexing: Slicing, Boolean Masking, Fancy Indexing")

grid = np.arange(1, 26).reshape(5, 5)
print(f"5x5 Source Grid:\n{grid}")

# 1. Multi-axis Slicing (Produces Views)
center_subgrid = grid[1:4, 1:4]
print(f"\nCenter 3x3 Subgrid (View):\n{center_subgrid}")

# Stepped Striding & Inversion
reversed_rows = grid[::-1, :]
print(f"Rows Reversed:\n{reversed_rows[:2]}...")

# 2. Boolean Masking & Conditional Filtering
# Find all elements divisible by 3 and greater than 10
mask = (grid % 3 == 0) & (grid > 10)
filtered_elements = grid[mask]
print(f"\nElements (divisible by 3 and > 10): {filtered_elements}")

# 3. Conditional Replacement via np.where
# Syntax: np.where(condition, value_if_true, value_if_false)
capped_grid = np.where(grid > 15, 99, grid)
print(f"\nGrid with values > 15 replaced with 99:\n{capped_grid}")

# 4. Multi-Condition Categorization via np.select
# Syntax: np.select([cond1, cond2], [choice1, choice2], default=default_val)
conditions = [
    grid < 10,
    (grid >= 10) & (grid <= 20),
    grid > 20
]
labels = [-1, 0, 1]  # Low, Medium, High codes
categorized = np.select(conditions, labels, default=999)
print(f"\nCategorized Grid (-1: <10, 0: 10-20, 1: >20):\n{categorized}")

# 5. Fancy / Integer Array Indexing (Produces COPIES, NOT Views)
# Selecting specific non-contiguous rows [0, 2, 4] and columns [1, 3]
row_indices = np.array([0, 2, 4])
col_indices = np.array([1, 3])
fancy_indexed = grid[row_indices[:, np.newaxis], col_indices]
print(f"\nFancy Indexed Submatrix (Rows [0, 2, 4], Cols [1, 3]):\n{fancy_indexed}")
print(f"Fancy indexing shares memory? {np.shares_memory(grid, fancy_indexed)} (Always False!)")


# ==============================================================================
# 6. SHAPE ENGINEERING, TRANSPOSITION, STACKING & SPLITTING
# ==============================================================================
print_section("6. Shape Engineering, Transposition, Stacking & Splitting")

cube = np.arange(24).reshape(2, 3, 4)
print(f"3D Cube Shape: {cube.shape}")

# Ravel vs. Flatten:
# - ravel(): Returns a flattened view whenever memory is contiguous (Fast, 0-copy)
# - flatten(): ALWAYS returns a deep copy
raveled = cube.ravel()
flattened = cube.flatten()
print(f"Ravel shares memory?   {np.shares_memory(cube, raveled)} (True -> View)")
print(f"Flatten shares memory? {np.shares_memory(cube, flattened)} (False -> Copy)")

# Dimension Expansion & Squeezing
vec = np.array([10, 20, 30])  # Shape: (3,)
col_vec = vec[:, np.newaxis]  # Shape: (3, 1)
row_vec = vec[np.newaxis, :]  # Shape: (1, 3)
expanded = np.expand_dims(vec, axis=(0, 2))  # Shape: (1, 3, 1)
squeezed = np.squeeze(expanded)              # Shape: (3,)

print(f"\nOriginal 1D shape:  {vec.shape}")
print(f"Column vector:      {col_vec.shape}")
print(f"Expanded dims (0,2):{expanded.shape}")
print(f"Squeezed back:      {squeezed.shape}")

# Matrix Transposition & Axis Swapping
# Swap axis 0 and 2 in the 3D cube (from (2, 3, 4) to (4, 3, 2))
transposed_cube = np.transpose(cube, axes=(2, 1, 0))
print(f"Transposed Cube Shape: {transposed_cube.shape}")

# Concatenation and Stacking
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vstacked = np.vstack((a, b))  # Vertical stack (axis 0)
hstacked = np.hstack((a, b))  # Horizontal stack (axis 1)
dstacked = np.dstack((a, b))  # Depth stack (axis 2)

print(f"\nVertical Stack (4, 2):\n{vstacked}")
print(f"Horizontal Stack (2, 4):\n{hstacked}")
print(f"Depth Stack (2, 2, 2) shape: {dstacked.shape}")

# Array Splitting
upper, lower = np.vsplit(vstacked, 2)
print(f"\nVertical Split Upper Half:\n{upper}")

print("\n[SUCCESS] Part 1 completed successfully.")
