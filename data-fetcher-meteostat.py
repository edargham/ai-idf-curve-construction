from datetime import datetime
from meteostat import Daily

# Set time period
start = datetime(2000, 6, 1)
end = datetime(2024, 6, 1)

beirut_rhia = '40100'

# Get daily data
data = Daily(beirut_rhia, start, end, model=False)
data = data.fetch()

# Keep only prcp column
data = data[['prcp']]

# Change the column name to 'value'
data = data.rename(columns={'prcp': 'value'})

# Change the index name to 'date'
data.index.name = 'date'

# Save data to CSV
data.to_csv('data/beirut-meteostat.csv', index=True)
print(data.tail())
print(data['value'].sum())