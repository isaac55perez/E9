# Project Context and Constraints

This document outlines the mathematical context and technical constraints of the convolution exercise.

## Mathematical Context

### Signal Processing with Vectors

The exercise demonstrates signal processing concepts using vector operations:

1. **Signal Vector**
   - Represents a continuous sine wave in discrete form
   - Uses 2000 samples (10 periods × 200 samples/period)
   - Each element represents signal amplitude at a specific time point

2. **Filter Vector**
   - 30-sample window extracted from the signal
   - Centered on the first peak of the sine wave
   - Acts as a pattern matching template

3. **Convolution Vector**
   - Result of convolving signal and filter vectors
   - Same length as input signal (2000 samples)
   - Represents correlation between filter and signal at each position

## Vector Operations

### Key Concepts

1. **Vector Indexing**
   - Zero-based indexing for all vectors
   - Supports positive and negative indices
   - Slice notation used for filter extraction

2. **Vector Arithmetic**
   - Element-wise operations in convolution calculation
   - Dot product implicit in convolution
   - Normalization not applied to preserve amplitude information

3. **Vector Broadcasting**
   - Used in signal generation with numpy operations
   - Ensures efficient computation of sine values

## Technical Constraints

1. **Memory Efficiency**
   - Vectors are stored as NumPy arrays
   - Contiguous memory allocation
   - Efficient broadcasting operations

2. **Computation Constraints**
   - Convolution performed in O(n log n) time using FFT
   - Peak detection uses local maxima algorithm
   - Vector operations vectorized for performance

3. **Numerical Precision**
   - Uses 64-bit floating-point numbers
   - Maintains numerical stability in convolution
   - Avoids accumulation of rounding errors

## Visualization Constraints

1. **Plot Range**
   - X-axis spans full signal length plus filter
   - Y-axis auto-scaled to data range
   - Filter shown at negative indices for clarity

2. **Resolution**
   - 200 samples per period ensures smooth visualization
   - Sufficient for accurate peak detection
   - Balances detail with computational efficiency

## Implementation Notes

- Vector operations prioritize NumPy's optimized methods
- In-place operations used where possible
- Memory usage scales linearly with signal length
- Peak detection parameters tuned for sine wave periodicity