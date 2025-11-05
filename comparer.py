import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from scipy import stats

def nash_sutcliffe_efficiency(observed, simulated):
  """
  Compute Nash-Sutcliffe Efficiency (NSE).

  Parameters:
    observed (array-like): Array of observed values.
    simulated (array-like): Array of simulated values.

  Returns:
    float: Nash-Sutcliffe Efficiency coefficient.
  """
  observed = np.array(observed)
  simulated = np.array(simulated)
  numerator = np.sum((observed - simulated) ** 2)
  denominator = np.sum((observed - np.mean(observed)) ** 2)
  return 1 - (numerator / denominator) if denominator != 0 else np.nan

# Load CSV files from the folder "data"
accumulated_path = "data-lib/gpm-imerg-bia-daily.csv"
meteostat_path = "data-lib/40100-refined.csv"

df_accumulated = pd.read_csv(accumulated_path)
df_meteostat = pd.read_csv(meteostat_path)

# Convert date columns to datetime
df_accumulated['date'] = pd.to_datetime(df_accumulated['date'])
df_meteostat['date'] = pd.to_datetime(df_meteostat['date'])

# Select rows from both datasets where the date column matches and both "value" columns are non-zero
merged_df = pd.merge(df_accumulated, df_meteostat, on='date', suffixes=('_acc', '_met'))
merged_df = merged_df.round(1)
merged_df = merged_df[(merged_df['value_met'] > 0)]
print(merged_df.head())

# Drop NaN rows
merged_df = merged_df.dropna()

# merged_df['value_met'] = MinMaxScaler().fit_transform(merged_df[['value_met']])

merged_df['bias_coef'] = merged_df['value_acc'] / (merged_df['value_met'] + 1e-4)
merged_df['diff'] = merged_df['value_acc'] - merged_df['value_met']

merged_df.to_csv('data/comp-merged.csv', index=False)

df_accumulated = merged_df[['date', 'value_acc']].rename(columns={'value_acc': 'value'})
df_meteostat = merged_df[['date', 'value_met']].rename(columns={'value_met': 'value'})

# To compare using RMSE, we need to ensure the two arrays align.
# For simplicity, we'll assume they align by index. If necessary, you can merge them on a common key.

# Align indices by taking the intersection of indices if lengths differ
common_index = df_accumulated.index.intersection(df_meteostat.index)
values_accumulated = merged_df[['date', 'value_acc']]
values_accumulated['value'] = values_accumulated['value_acc']
values_accumulated = values_accumulated.drop(columns='value_acc')
print(values_accumulated)
values_accumulated.to_csv('data/comp-accumulated.csv', index=False)

values_meteostat = merged_df[['date', 'value_met']]
values_meteostat['value'] = values_meteostat['value_met']
values_meteostat = values_meteostat.drop(columns='value_met')
print(values_meteostat)
values_meteostat.to_csv('data/comp-meteostat.csv', index=False)

# Compute RMSE
mse = np.mean((values_accumulated['value'] - values_meteostat['value']) ** 2)
rmse = sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)

# Compare the sum of the values
sum_accumulated = values_accumulated['value'].sum()
sum_meteostat = values_meteostat['value'].sum()
print("Sum of accumulated values:", sum_accumulated)
print("Sum of meteostat values:", sum_meteostat)
print("Difference in sums:", abs(sum_accumulated - sum_meteostat))

# Compute R2 score
r2 = r2_score(
  values_meteostat['value'],
  values_accumulated['value'],
)
print("R2 score:", r2)

# Compute correlation coefficient
correlation = values_accumulated['value'].corr(values_meteostat['value'])
print("Correlation coefficient:", correlation)

# Compute MAE
mae = np.mean(np.abs(values_accumulated['value'] - values_meteostat['value']))
print("MAE:", mae)

# Compute NSE
nse = nash_sutcliffe_efficiency(
  values_accumulated['value'],
  values_meteostat['value'],
)
print("NSE:", nse)

print("\n -- More Stats -- \n")

# Calculate standard deviation of each dataset
std_acc = merged_df['value_acc'].std()
std_met = merged_df['value_met'].std()

mean_acc = merged_df['value_acc'].mean()
mean_met = merged_df['value_met'].mean()

# Perform a paired t-test (if data is normally distributed) or Wilcoxon signed-rank test
# Here, we use Wilcoxon due to potential non-normality
stat, p = stats.wilcoxon(merged_df['value_met'], merged_df['value_acc'])

print(f"Mean of value_acc: {mean_acc}, value_met: {mean_met}")
print(f"Standard Deviation of value_acc: {std_acc}, value_met: {std_met}")
print(f"Wilcoxon Test Statistic: {stat}, p-value: {p}")

# Interpret p-value
if p < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")