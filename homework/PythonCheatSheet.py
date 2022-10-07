# Useful Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

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

# np.linalg.inv(matrix): matrix inverse
inverse = np.linalg.inv(matrix_1)

# np.ones(n): creates a nx1 matrix of 1's
ones = np.ones(10)

# shape(): returns the number of rows and columns as a tuple. Ex: (2, 4)
df.shape()

# sort_values(): sorts the values
# optional argument 1: says what column to sort on
# sort descending with ascending=False
df.sort_values()
df.sort_values(df.columns[0])
df.sort_values(ascending=False)

# skew(): calculates skewness of data set
df.skew()

# kurtosis(): calculates excess kurtosis (how fat are your tails?)
df.kurtosis()

# quantile(q): returns values at q-th quantile
df.quantile(0.05)

# cumprod(): returns cumulative product
df.cumprod()

# cummax(): returns cumulative maximum
df.cummax()

# pd.DataFrame(): creates a data frame
x = 1
y = 2
z = 3
pd.DataFrame(data = [x, y, z], 
    index = ['X Value', 'Y Value', 'Z Value'], 
    columns = ['Example Values'])

################# Mathematics #################
# Square Root
np.sqrt(12)

################# Useful Methods #################
# This method takes return data and portfolio weights, and returns mean, volatitilty and Sharpe
def portfolio_stats(excessReturnData, portfolio_weights):
    # Calculate the mean by multiplying the mean excess returns by the tangency weights and annualizing
    # TODO: double check where these formulas came from (class notes?)
    mean = excessReturnData.mean() @ portfolio_weights * 12

    # Volatility = sqrt(variance), and by class notes: variance = allocation_matrix * covariance_matrix * allocation_matrix
    # Annualize the result with sqrt(12)
    vol = np.sqrt(portfolio_weights @ excessReturnData.cov() @ portfolio_weights) * np.sqrt(12)

    # Sharpe Ratio is mean / vol
    sharpe_ratio = mean / vol

    # Format for easy reading
    return round(pd.DataFrame(data = [mean, vol, sharpe_ratio], 
        index = ['Mean', 'Volatility', 'Sharpe'], 
        columns = ['Portfolio Stats']), 4)
    
# This method takes returns data and returns mean, volatility and Sharpe
def portfolio_stats_2(data):
    # Calculate the mean and annualize
    mean = data.mean() * 12

    # Volatility = standard deviation
    # Annualize the result with sqrt(12)
    vol = data.std() * np.sqrt(12)

    # Sharpe Ratio is mean / vol
    sharpe_ratio = mean / vol

    # Format for easy reading
    return round(pd.DataFrame(data = [mean, vol, sharpe_ratio], 
        index = ['Mean', 'Volatility', 'Sharpe']), 4)
    
# This method calculates regression statistics from OLS like beta, Treynor Ratio, and Information Ratio
# Beta is the beta slope from Ordinary Least Squares, and shows how your portfolio moves for every dollar the market moves (lower = less risk in case of downturn)
# Treynor Ratio is a measure of return based on systematic risk (like Sharpe but based on beta instead) (higher = better)
#       Treynor = (portfolio risk - risk free rate)/beta
# Information Ratio is the measure of returns beyond a benchmark compared to those returns' volatility (higher = better)
#       Information Ratio = (portfolio return - benchmark return)/(tracking error), where tracking error = standard deviation of excess return
def regression_stats(df):
    reg_stats = pd.DataFrame(data = None, index = df.columns, columns = ['beta', 
                                                                         'Treynor Ratio', 
                                                                         'Information Ratio'])
    for col in df.columns:
        # Drop the NAs in y
        y = df[col].dropna()
        # Align the X with y - this is us including the intercept
        X = sm.add_constant(factor_data['SPY US Equity'].loc[y.index])
        reg = sm.OLS(y, X).fit()
        reg_stats.loc[col, 'beta'] = reg.params[1]
        # Treynor is calulated as mean/beta
        reg_stats.loc[col, 'Treynor Ratio'] = (df[col].mean() * 12) / reg.params[1]
        # Information Ratio = (portfolio return - benchmark return)/(tracking error), annualized by sqrt(12)
        reg_stats.loc[col, 'Information Ratio'] = (reg.params[0] / reg.resid.std()) * np.sqrt(12)

    return reg_stats.astype(float).round(4)