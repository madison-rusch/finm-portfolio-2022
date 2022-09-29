# Useful Libraries
import pandas as pd
import numpy as np

################# Data ingestion #################
# read_excel(path): reads in excel file at path
# sheet_name='something' lets you import the given sheet (this is an optional argument)
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

# unstack(): pivots columns and groups them as secondary rows by original row
#    ex:       A   B   C
#          A   1   2   3   
#          B   4   5   6
#          C   7   8   9
#    becomes:
#          A   A   1
#              B   2   
#              C   3
#          B   A   4
#              B   5   
#              C   6
#          C   A   7
#              B   8   
#              C   9
df.unstack()

# cov(): returns matrix with covariance between columns
df.cov()

# np.mat(): converts input to a matrix
matrix_1 = np.mat([[1,2], [3,4]])
matrix_2 = np.mat([[1,2], [7,8]])

# np.matmul(A,B): matrix multiplication (equivalent to @ symbol)
result = np.matmul(matrix_1, matrix_2)

# @ symbol: used after Python v3.5 for Matrix multiplication (equivalent to np.matmul(A, B))
# --NOTE-- this is also used for property decorators (like in Seb's class)
result = matrix_1 @ matrix_2

################# Mathematics #################
# Square Root
np.sqrt(12)