import pandas as pd
import numpy as np

def build_train_val(df, train_years=None, val_years=None):
    """Build train/validation DataFrames per-duration using year-based split.

    Returns: (train_df_combined, val_df_combined, years)
    """
    if train_years is None:
        train_years = list(range(1998, 2019))
    if val_years is None:
        val_years = list(range(2019, 2026))

    # Attempt to detect year column
    if 'year' in df.columns:
        years = df['year'].astype(int)
    else:
        # fallback: try first column if it looks like year
        first_col = df.columns[0]
        try:
            years = df[first_col].astype(int)
        except Exception:
            # last resort: use dataframe index
            try:
                years = df.index.astype(int)
            except Exception:
                raise RuntimeError('Could not determine year column in annual_max_intensity.csv')

    def rank_data(data):
        n = len(data)
        ranks = np.arange(1, n + 1)
        return ranks / (n + 1)

    duration_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]
    col_names = ['5mns','10mns','15mns','30mns','1h','90min','2h','3h','6h','12h','15h','18h','24h']

    train_dfs = []
    val_dfs = []
    for col, dmin in zip(col_names, duration_minutes):
        series = df[col]
        # align series with years
        ser_with_year = pd.Series(series.values, index=years)
        train_ser = ser_with_year[ser_with_year.index.isin(train_years)].dropna()
        val_ser = ser_with_year[ser_with_year.index.isin(val_years)].dropna()

        if len(train_ser) > 0:
            train_dfs.append(pd.DataFrame({
                'duration': dmin,
                'intensity': train_ser.values,
                'weibull_rank': rank_data(train_ser.values)
            }))

        if len(val_ser) > 0:
            val_dfs.append(pd.DataFrame({
                'duration': dmin,
                'intensity': val_ser.values,
                'weibull_rank': rank_data(val_ser.values)
            }))

    if len(train_dfs) == 0:
        raise RuntimeError('No training data found for years 1998-2018')

    train_df_combined = pd.concat(train_dfs, ignore_index=True)
    val_df_combined = pd.concat(val_dfs, ignore_index=True) if len(val_dfs) > 0 else pd.DataFrame(columns=['duration','intensity','weibull_rank'])

    return train_df_combined, val_df_combined, years
    