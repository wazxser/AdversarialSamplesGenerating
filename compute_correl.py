import pandas as pd
import numpy as np
import scipy.stats as stats

csv_data = pd.read_csv('compute_correl.csv', header=None)

csv_data = np.array(csv_data, dtype=float).reshape(2, 30)
print(csv_data)

print(stats.pearsonr(csv_data[0], csv_data[1]))
print(stats.spearmanr(csv_data[0], csv_data[1]))
print(stats.kendalltau(csv_data[0], csv_data[1]))