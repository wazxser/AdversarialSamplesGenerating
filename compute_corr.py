from scipy.stats import pearsonr
from scipy import stats
import csv
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import r2_score

csvfile = open('result_robust_nc_2.csv', 'r')
reader = csv.reader(csvfile)
i = 0
x = [0] * 30
y = [0] * 30
a, b = 0, 0
for item in reader:
    if i % 3 == 1:
        x[a] = float(item[2])
        a += 1
    elif i % 3 == 2:
        y[b] = float(item[2])
        b += 1
    i += 1
    if i == 90:
        break
x_scaled = preprocessing.scale(x)
y_scaled = preprocessing.scale(y)

print(pearsonr(x, y))
print(pearsonr(x_scaled, y_scaled))
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print(r_value)
print(r_value*r_value)
# import pandas as pd
#
# df = pd.DataFrame({'A': [0.101132813, 0.107679688, 0.114753906, 0.116179688, 0.117125, 0.114222656, 0.113539063],
#                    'B': [0.804391892, 0.800310811, 0.799554054, 0.799797297, 0.799202703, 0.800081081, 0.798594595]})
#
# # df['a'] = [0.101132813, 0.107679688, 0.114753906, 0.11617968, 0.117125, 0.114222656, 0.113539063]
# # df['b'] = [0.80439182, 0.800310811, 0.799554054, 0.799797297, 0.799202703, 0.800081081, 0.798594595]
#
# print(df.corr())