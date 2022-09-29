# Useful Libraries
import pandas as pd
import numpy as np

################# Data ingestion #################
df = pd.read_excel('path_to_excel_workbook', sheet_name='sheet_name')

################# Statistics Calculations #################

# mean(): calculates mean of columns and pivots column titles to index if you use to_frame()
# axis=1 allows you to calculate the mean of a row
df.mean()
df.mean(axis=1)
df.mean().to_frame('Mean')

# std(): calculates the standard deviation of the columns
# skipna = True prevents NaN errors
df.std()
df.std(skipna=True)

# nlargest(): gets largest n values from given column
df.nlargest(1, 'Column_Name')

# nsmallest(): gets smallest n values from given column
df.nsmallest(1, 'Column_Name')

# corr(): calculates the correlation matrix
df.corr()


################# Mathematics #################
# Square Root
np.sqrt(12)