import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1729) # Hardy-Ramanujan number

# --- Core Function: Spectral Coherence Coefficient C_N ---
def compute_C_stats(s, Ns):
    """
    Computes mean and variance of C_N for a sequence s over various window sizes N.
    s: array of normalized gaps (stationary, mean ~ 1)
    Ns: list of window sizes
    """
    stats = []
    for N in Ns:
        c_values = []
        # Use a stride to reduce correlation between windows
        stride = max(1, N // 2)
        for i in range(0, len(s) - N, stride):
            window = s[i : i+N]
            num = np.sum(window[:-1]) # Sum of first N-1
            den = np.sum(window)      # Sum of all N
            if den > 0:
                c_values.append(num / den)
        
        c_values = np.array(c_values)
        if len(c_values) > 0:
            mean_c = np.mean(c_values)
            var_c = np.var(c_values)
            stats.append({'N': N, 'mean': mean_c, 'var': var_c, 'count': len(c_values)})
    
    return pd.DataFrame(stats)

# --- Simulation of L-Function Zeros ---

# 1. Algebraic/Consistent Regime (r_an = r_alg)
# Modeled by GUE-like rigidity (short-range correlations).
# Corresponds to a valid modular L-function.
def generate_algebraic_gaps(n_gaps):
    # AR(1) model with negative correlation to mimic level repulsion (GUE)
    phi = -0.36 
    noise = np.random.normal(1, 0.3, n_gaps) 
    gaps = np.zeros(n_gaps); gaps[0] = 1.0
    for t in range(1, n_gaps):
        gaps[t] = 1.0 + phi * (gaps[t-1] - 1.0) + (noise[t] - 1.0)
    gaps = np.maximum(gaps, 0.01)
    gaps = gaps / np.mean(gaps)
    return gaps

# 2. Ghost Zero/Mismatch Regime (r_an > r_alg)
# Modeled by a spectrum with clustering/long-range correlations.
# Simulates the "spectral disorder" caused by zeros not supported by geometry.
def generate_ghost_gaps(n_gaps):
    # 1/f noise (Pink noise) to simulate lack of rigidity/clustering
    white = np.random.normal(0, 1, n_gaps)
    freqs = np.fft.rfftfreq(n_gaps)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 1e-10)); scale[0] = 0
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n_gaps)
    # Exponentiate to get positive gaps with clustering (small gaps)
    gaps = np.exp(pink)
    gaps = gaps / np.mean(gaps)
    return gaps

# --- Main Execution ---
n_gaps = 200000
window_sizes = [5, 10, 20, 40, 80, 160, 320]

print("Generating Algebraic Spectrum...")
s_alg = generate_algebraic_gaps(n_gaps)
print("Generating Ghost Zero Spectrum...")
s_ghost = generate_ghost_gaps(n_gaps)

print("Computing statistics...")
df_alg = compute_C_stats(s_alg, window_sizes)
df_ghost = compute_C_stats(s_ghost, window_sizes)

# Theoretical Mean
df_alg['theory_mean'] = (df_alg['N'] - 1) / df_alg['N']

# --- Plotting ---

# Figure BSD1: Mean C_N vs N (Algebraic)
plt.figure(figsize=(8, 5))
plt.plot(df_alg['N'], df_alg['mean'], 'o-', label='Simulated (Algebraic Rank)', color='blue')
plt.plot(df_alg['N'], df_alg['theory_mean'], 'x--', label='Theory (N-1)/N', color='red')
plt.xlabel('Window Size N (Zeros)')
plt.ylabel('Mean Coherence $E[C_N]$')
plt.title('BSD1. Mean Spectral Coherence vs N (Consistent Rank)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xscale('log')
plt.savefig('Fig_BSD1_Mean.png')

# Figure BSD2: Variance vs N (Algebraic)
plt.figure(figsize=(8, 5))
plt.loglog(df_alg['N'], df_alg['var'], 'o-', label='Simulated (Algebraic Rank)', color='blue')
# Reference line N^-2
ref_x = np.array(window_sizes)
ref_y = df_alg['var'].iloc[0] * (ref_x[0] / ref_x)**2
plt.loglog(ref_x, ref_y, 'k--', label='Reference $N^{-2}$ (Rigid)')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('BSD2. Variance of Coherence (Geometric Stability)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_BSD2_Variance.png')

# Figure BSD3: Stress Test (Algebraic vs Ghost)
plt.figure(figsize=(8, 5))
plt.loglog(df_alg['N'], df_alg['var'], 'o-', label='Algebraic (Consistent)', color='blue')
plt.loglog(df_ghost['N'], df_ghost['var'], 's-', label='Ghost Zeros (Mismatch)', color='red')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('BSD3. Bridge A: Detection of Rank Mismatch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_BSD3_StressTest.png')

# Output data for Table
print("\nAlgebraic Statistics:")
print(df_alg[['N', 'mean', 'var']])
print("\nGhost Zero Statistics:")
print(df_ghost[['N', 'mean', 'var']])