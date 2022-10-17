# Useful Libraries
from re import L
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy

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

# Example correlation between two assets
regression_data.corr().loc['IWM US Equity', 'SPY US Equity']


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

# add_constant(): adds a value of 1 to every item in the data passed - in a dataframe, this might add a column of ones
#   generally used in replication to calculate OLS with an intercept
sm.add_constant(df)

# OLS(): used to replicate portfolios, or more specifically run OLS analysis
# params: how you can get the values of the intercept and betas
# rsquared: R-squared
#       Example from HW 2:
#           # Replicate HFRI with the merrill_factors data
#           model = sm.OLS(HFRI, HFRI_with_constant).fit()
#           # Report the intercept and betas
#           model.params.to_frame('Regression Parameters')
#           # Get R-squared
#           round(model.rsquared, 4)
sm.OLS(y, X).fit()

# Regression with two assets:
#       y = regression_data['EEM US Equity']
#       X = regression_data['SPY US Equity']

#       hedge_reg = sm.OLS(y, X).fit()

# In this example, Beta is hedge_reg.params[0]


# pd.DataFrame(): creates a data frame
x = 1
y = 2
z = 3
pd.DataFrame(data = [x, y, z], 
    index = ['X Value', 'Y Value', 'Z Value'], 
    columns = ['Example Values'])

# To Calculate Expanding VaR, Basic VaR, and CVaR
returns = pd.DataFrame()
historic_var = returns.expanding(60).quantile(0.05)    
VaR = returns.quantile(0.05)
CVaR = (returns[returns < returns.quantile(0.05)]).mean()

# To Plot
historic_var.plot(kind='line')

# Expanding Vol (eventually for VaR)
sigma_expanding = returns.expanding(60).std()
sigma_expanding.plot()

# Rolling Vol (eventually for VaR)
sigma_rolling = returns.rolling(60).std()
sigma_rolling.plot()

# Log Returns: use the method to get the log returns of an asset. NOTE: the 1 + is to account for cumulative returns
np.log(1+df['SPY US Equity'])

# Calculating VaR (assuming Normal)
volatility = -.0009
q = 0.01
mu = 0
z_phi = scipy.stats.norm.ppf(q)
VaR_estimate = mu + z_phi*volatility

# Calculating CVaR (assuming Normal)
q = 0.05
z = scipy.stats.norm.ppf(q)
coef_CVaR = -stats.norm().cdf(z)/q
CVaR_estimate = mu + coef_CVaR*volatility

# Example of subtracting a risk free rate from a dataframe
df_ex = df.subtract(df['USGG3M Index'],axis=0).drop(columns=['USGG3M Index'])

# corrcoeff: can be used to find the correlation coefficient of two datasets (see HW 2 out-of-sample replication)
np.corrcoef(df1, df2)

# model.params will give you all Beta values of the regression after running OLS
model.params

################# Mathematics #################
# Square Root
np.sqrt(12)

################# Useful Methods #################
# This method takes return data and portfolio weights, and returns mean, volatitilty and Sharpe
# Good for tangency portfolios, optimal portfolios, and out of sample optimal portfolios (HW 1)
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
# Information Ratio is the measure of returns beyond a benchmark compared to those returns' volatility (higher = better). Also known as Sharpe Ratio of Hedged position
#       Information Ratio = (portfolio return - benchmark return)/(tracking error), where tracking error = standard deviation of excess return
def regression_stats(df):
    reg_stats = pd.DataFrame(data = None, index = df.columns, columns = ['beta', 
                                                                         'Treynor Ratio', 
                                                                         'Information Ratio'])
    for col in df.columns:
        # Drop the NAs in y
        y = df[col].dropna()
        # Align the X with y - this is us including the intercept
        X = sm.add_constant(df['SPY US Equity'].loc[y.index])
        reg = sm.OLS(y, X).fit()
        reg_stats.loc[col, 'beta'] = reg.params[1]
        # Treynor is calulated as mean/beta
        reg_stats.loc[col, 'Treynor Ratio'] = (df[col].mean() * 12) / reg.params[1]
        # Information Ratio = (portfolio return - benchmark return)/(tracking error), annualized by sqrt(12)
        # also calculated as alpha/standard deviation of residuals
        reg_stats.loc[col, 'Information Ratio'] = (reg.params[0] / reg.resid.std()) * np.sqrt(12)

    return reg_stats.astype(float).round(4)

# This method works to calculate Maximum Drawdown
def maximumDrawdown(returns):
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        max_drawdown = drawdown.min()
        end_date = drawdown.idxmin()
        summary = pd.DataFrame({'Max Drawdown': max_drawdown, 'Bottom': end_date})

        for col in drawdown:
            summary.loc[col,'Peak'] = (rolling_max.loc[:end_date[col],col]).idxmax()
            recovery = (drawdown.loc[end_date[col]:,col])
            try:
                summary.loc[col,'Recover'] = pd.to_datetime(recovery[recovery >= 0].index[0])
            except:
                summary.loc[col,'Recover'] = pd.to_datetime(None)

            summary['Peak'] = pd.to_datetime(summary['Peak'])
            try:
                summary['Duration (to Recover)'] = (summary['Recover'] - summary['Peak'])
            except:
                summary['Duration (to Recover)'] = None

            summary = summary[['Max Drawdown','Peak','Bottom','Recover','Duration (to Recover)']]

        return summary  

# This method calculates VaR and CVaR for a given portfolio (also skewness and kurtosis)
def tailMetrics(returns, quantile=.05, relative=False, mdd=True):    
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurtosis()

    VaR = returns.quantile(quantile)
    CVaR = (returns[returns < returns.quantile(quantile)]).mean()

    if relative:
        VaR = (VaR - returns.mean())/returns.std()
        CVaR = (CVaR - returns.mean())/returns.std()

    metrics[f'VaR ({quantile})'] = VaR
    metrics[f'CVaR ({quantile})'] = CVaR

    if mdd:
        mdd_stats = maximumDrawdown(returns)
        metrics = metrics.join(mdd_stats)

        if relative:
            metrics['Max Drawdown'] = (metrics['Max Drawdown'] - returns.mean())/returns.std()

    return metrics

# calculates probability of underperformance compared to the benchmark (risk-free rate in Barnes example)
# c from the equations is what we're comparing against, so for risk free rate c = 0, and for if we can exceed 6%, use c = 0.06
import scipy.stats as stats
def prob_calc(h, tilde_mu, tilde_sigma):
    return stats.norm.cdf((- np.sqrt(h) * tilde_mu) / tilde_sigma)

# This method is used to generate a correlation heatmap and optionally provide the MIN and MAX correlation pair
# See Mark's ProShares discussion for example usage (https://github.com/MarkHendricks/finm-portfolio-2022/blob/main/discussions/Case%202%20-%20ProShares%20ETF.ipynb)
def display_correlation(df,annot=True,list_maxmin=True):
    
    corrmat = df.corr()
    #ignore self-correlation
    corrmat[corrmat==1] = None
    sns.heatmap(corrmat,annot=annot)

    if list_maxmin:
        corr_rank = corrmat.unstack().sort_values().dropna()
        pair_max = corr_rank.index[-1]
        pair_min = corr_rank.index[0]

        print(f'MIN Correlation pair is {pair_min}')
        print(f'MAX Correlation pair is {pair_max}')
        
    return

# Example of Replication with and without an intercept
rep_spy = df[['SPY US Equity']].copy()

model = sm.OLS(df['SPY US Equity'],sm.add_constant(df.drop(columns=['SPY US Equity'])))
rep_spy['Static-IS-Int'] = model.fit().fittedvalues
model = sm.OLS(df['SPY US Equity'],df.drop(columns=['SPY US Equity']))
rep_spy['Static-IS-NoInt'] = model.fit().fittedvalues
portfolio_stats_2(rep_spy)

# Compute the weights of the tangency portfolio
def compute_tangency(excessReturnMatrix):
    # Get the covariance matrix based on excess returns
    sigma = excessReturnMatrix.cov()
    
    # Get the number of asset classes (in this example should be 11)
    n = sigma.shape[0]
    
    # Get the vector of mean excess returns
    mu = excessReturnMatrix.mean()
    
    # Get sigma inverse
    sigma_inv = np.linalg.inv(sigma)
    
    # Now we have all the pieces, do the calculation
    weights = (sigma_inv @ mu) / (np.ones(n) @ sigma_inv @ mu)
    
    # Convert back to a Series for convenience
    return pd.Series(weights, index=mu.index)

# Compute the optimal portfolio given a target mean return
def target_mv_portfolio(df_tilde, target_return=0.01, diagonalize_Sigma=False):

    omega_tangency, mu_tilde, Sigma = compute_tangency(df_tilde, diagonalize_Sigma=diagonalize_Sigma)

    Sigma_adj = Sigma.copy()

    if diagonalize_Sigma:

        Sigma_adj.loc[:,:] = np.diag(np.diag(Sigma_adj))

    Sigma_inv = np.linalg.inv(Sigma_adj)

    N = Sigma_adj.shape[0]

    delta_tilde = ((np.ones(N) @ Sigma_inv @ mu_tilde)/(mu_tilde @ Sigma_inv @ mu_tilde)) * target_return

    omega_star = delta_tilde * omega_tangency

    return omega_star, mu_tilde, Sigma_adj

################# Midterm Notes #################
'''
MV Optimization (Tangency Portfolios)

 - True or False? Mean-variance optimization goes long the highest Sharpe-Ratio assets and shorts the lowest Sharpe-ratio assets.
        False. MV Optimization seeks to maximize Sharpe of the portfolio, but that is not achieved by weighting individual assets
        proportional to their individual Sharpe ratios. Rather, an asset's covariances are an important determinant in whether it 
        has a high/low, positive/negative weight.
        
 - True or False? The Tangency portfolio weights assets in proportion to their Sharpe ratios.
        False. Weights account for covariances, not just volatilities. Weights are determined based on the solution of optimization
        problem where we try to minimize the covariance matrix.
        
 - True or False? Suppose we have k risky securities, and an equally weighted portfolio is formed from them. If pairwise correlations 
   across k security returns are less than perfect, an equally weighted portfolio becomes riskless as k approaches infinity.
        False. Portfolio would become riskless only if pairwise correlations across k security returns were zero. In our case, it's
        not zero, that's why portfolio isn't riskless as k approaches infinity. i.e. riskless = no correlation between assets
        
- If you need to zero out everything but the variances (like in Midterm 2020), do this:
 for i in Sigma:
     for j in Sigma:
         if i != j:
             Sigma[i][j] = 0
            
 - MV Optimization fails out-of-sample for two reasons:
    1. Imprecise estimation of covariance matrix: The covariance matrix is poorly estimated in the case of large number of assets or less amount 
        of historical data. Inverting the covariance matrix makes the estimation even more fragile. Inverting a matrix with high correlations increases 
        the condition number further adding to the instability. Due to these, our estimates of covariances will likely not hold out-of-sample.
    2. High senstivity to changes in mean return: MV optimizer is highly sensitive to small changes in the estimated mean returns of the security pool.
        Large swings in portfolio weights are required to maintain the optimal portfolio even with small changes in mean returns. Due to this, MV 
        optimizer does not perform well on out-of-sample data   

'''

'''
Regression/Replication

 - Short Answer. Suppose we have a security, r. Explain how to construct its information ratio with respect to a benchmark of z.
    1. First run regression with an intercept
    2. Calculate the standard deviation of the residuals.
    3. Use the formula IR = alpha/epsilon 
    - See def regression_stats(df)

 - In hedging, short the beta of the regression. If replicating, long the beta of the replication/regression

 - If you want to find the mean of a hedged portfolio, use mu_hedged = mu_EEM - beta * mu_SPY (2021 Exam, Q3.3)

 - Sharpe Ratio: mean/vol. How much return are you getting for your level of risk? Always use excess returns unless stated otherwise
 - Treynor Ratio: Sharpe ratio of betas (mean excess return/beta). It is a representation of how much return was generated for each unit of risk taken
                  on by the portfolio. Risk here generally refers to systematic risk. For the same level of exposure to the system (beta) Treynor will
                  be higher when returns are higher. Lets us normalize returns based on exposure to the market
 - Information Ratio: Sharpe Ratio of hedged position (alpha/sigma epsilon). This is the tradeoff between alpha and the unexplained volatility.
                      In the ProShares example, the indexes had positive returns, but when replicated (hedged) with SPY, we see the info ratios
                      go negative. They drastically underperformed SPY. Is your alpha actually due to smart decisions, or is it due to volatility in the 
                      market? Higher IR = more attribution that your returns are due to smart decisions
                      
 - Skewness is important for Hedge Funds, because if they're hedging properly we expect them to be less skewed than the market
 
 - When replicating, matching returns is more important, so don't include an intercept (This is what Merrill Lynch did)
 - When hedging, matching variance is more important, so include an intercept. This makes the betas capture the variability/variance better
'''

'''
Modeling Volatility and VaR
 - Historic VaR: expanding based on the first amount of time and only adding on more from there.
    - Drawbacks:
        ~ Backward Looking: looks only at historical data
        ~ Ghosting Effect: equal weight given to far away observations i.e. 1920 weighted same as 2020
        ~ Slow to React: due to Ghosting, less weight is given to most recent observations
 
Log Returns:
 - To get log returns of an asset, use np.log(1+modeling_risk_data['SPY US Equity'])
 
The probability that the cumulative Market Returns < cumulative Risk Free Returns, use normal cdf, c=0, (c-mu)/sigma

 - Drawbacks of Historical VaR
    - Backward looking: only looks at historical data
    - Ghosting Effect: Equal weights on observations that are very far away time-wise
    - Slow to React: Because of ghosting, it puts less weight on more recent observations
    
Probability of Underperformance can be calculated with CDF, using log returns:
    cdf((-sqrt(h)*mu)/sigma)
    mu = returns
    sigma = vol
'''