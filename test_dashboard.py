import pandas as pd
import datapilot as dp
import numpy as np

# Create dummy data
df = pd.DataFrame({
    'Age': np.random.randint(18, 60, 100),
    'Salary': np.random.normal(50000, 10000, 100),
    'MissingCol': [np.nan if i < 20 else i for i in range(100)],
    'Category': ['A', 'B', 'C', 'D'] * 25
})

dp.dashboard(df, output_path="test_dashboard.html")
print("Dashboard generated successfully.")
