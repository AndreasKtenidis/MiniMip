# class PropensityScore(StatisticalFunction):
#
#     def compute(self, *args, **kwargs):
#         pass

import pandas as pd
import numpy as np

# Example data
data = pd.DataFrame({
    'ID': range(1, 11),
    'Treated': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'propensity_score': [0.65, 0.67, 0.72, 0.75, 0.80, 0.64, 0.68, 0.74, 0.78, 0.82]
})

print(data)


# Create histogram bins for propensity scores
bins = np.linspace(0.6, 0.85, 4)  # e.g., bins: [0.6–0.683), [0.683–0.767), [0.767–0.85]
data['bin'] = pd.cut(data['propensity_score'], bins=bins, include_lowest=True)

print("\nData with Histogram Bins:")
print(data[['ID', 'Treated', 'propensity_score', 'bin']])



# Count treated and control units per bin
bin_summary = data.groupby(['bin', 'Treated'], observed=False).size().unstack(fill_value=0)
bin_summary.columns = ['Control_Count', 'Treated_Count']  # 0=Control, 1=Treated
print("\nBin Summary:")
print(bin_summary)

# Find treated and control units in each bin
matched_pairs = []

for bin_interval in data['bin'].unique():
    treated_units = data[(data['bin'] == bin_interval) & (data['Treated'] == 1)]
    control_units = data[(data['bin'] == bin_interval) & (data['Treated'] == 0)]

    # Pair up based on min(len(treated), len(control)) within each bin
    min_pairs = min(len(treated_units), len(control_units))

    for i in range(min_pairs):
        matched_pairs.append({
            'Treated_ID': treated_units.iloc[i]['ID'],
            'Control_ID': control_units.iloc[i]['ID'],
            'Bin': bin_interval
        })

matched_df = pd.DataFrame(matched_pairs)
print("\nMatched Pairs by Histogram Bins:")
print(matched_df)
