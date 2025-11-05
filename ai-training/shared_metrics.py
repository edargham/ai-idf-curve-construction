import numpy as np

# Calculate Nash-Sutcliffe Efficiency (NSE)
def nash_sutcliffe_efficiency(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan

def squared_pearson_r2(observed, simulated):
    """Calculate squared Pearson correlation coefficient (R²) between observed and simulated values."""
    observed = np.array(observed)
    simulated = np.array(simulated)
    if len(observed) != len(simulated) or len(observed) == 0:
        return np.nan
    correlation_matrix = np.corrcoef(observed, simulated)
    if correlation_matrix.shape != (2, 2):
        return np.nan
    r = correlation_matrix[0, 1]
    return r ** 2