import pandas as pd
import numpy as np
import datapilot as dp

df = pd.DataFrame({
    'Age': np.random.randint(18, 60, 100),
    'Salary': np.random.normal(50000, 10000, 100),
    'Category': ['A', 'B', 'C', 'D'] * 25
})

print("Testing histogram...")
fig1 = dp.hist(df, column='Age', hue='Category')

print("Testing scatter...")
fig2 = dp.scatter(df, x='Age', y='Salary', trendline=True)

print("Testing box...")
fig3 = dp.box(df, column='Salary', group_by='Category')

print("Testing violin...")
fig4 = dp.violin(df, column='Salary', group_by='Category')

print("Testing heatmap...")
fig5 = dp.heatmap(df)

# Note: We won't test visualize_ai because it calls LLM locally and might need keys
# but it uses the same px.* functions.

print("All tests generated successfully! (Figures were not shown to avoid blocking the script)")
