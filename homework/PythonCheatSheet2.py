import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame()

# To plot the cumulative returns of a dataset of excess returns
((df + 1).cumprod() - 1).plot(title = "Full-sample")

# transpose(): used to switch the rows and the columns of a dataframe
df.transpose()

# Gets univariate stats like mean, volatility, Sharpe and VaR for data
def portfolio_stats(data):
    # Calculate the mean and annualize
    mean = data.mean() * 12

    # Volatility = standard deviation
    # Annualize the result with sqrt(12)
    vol = data.std() * np.sqrt(12)

    # Sharpe Ratio is mean / vol
    sharpe_ratio = mean / vol
    
    # VaR
    var = data.quantile(0.05)

    # Format for easy reading
    return round(pd.DataFrame(data = [mean, vol, sharpe_ratio, var], 
        index = ['Mean', 'Volatility', 'Sharpe', 'VaR (0.05)']), 4)

# Gets alpha, beta, Treynor, and Info ratio for dataset (See HW 4)
for portf in portfolioData_excessReturns.columns:
    lhs = capm_data[portf]
    res = sm.OLS(lhs, riskFreeRegressor, missing='drop').fit()
    capm_report.loc[portf, 'alpha'] = res.params['const'] * 12
    capm_report.loc[portf, 'beta'] = res.params['Mkt-RF']
    capm_report.loc[portf, 'Information Ratio'] = np.sqrt(12) * res.params['const'] / res.resid.std()
    capm_report.loc[portf, 'Treynor Ratio'] = 12 * capm_data[portf].mean() / res.params['Mkt-RF']
    bm_residuals[portf] = res.resid
    t_p_values.loc[portf, 't-value'] = res.params['const']
    t_p_values.loc[portf, 't-value'] = res.tvalues['const']
    t_p_values.loc[portf, 'p-value'] = round(res.pvalues['const'], 4)

# Useful method for running a time series test on a factor model
# Returns alphas and R-squared values
# df - the portfolio excess returns data
# factor_df - the factor excess returns data
# factors - a list of factors we're testing (ex: AQR = ['MKT', 'HML', 'RMW', 'UMD'])
# test - label for test (ex: 'AQR')
def time_series_test(df, factor_df, factors, test, annualization=12):
    res = pd.DataFrame(data = None, index = df.columns, columns = [test + ' alpha', test + ' R-squared'])
    
    for port in df.columns:
        y = df[port]
        X = sm.add_constant(factor_df[factors])
        model = sm.OLS(y, X).fit()
        res.loc[port] = [model.params[0] * annualization, model.rsquared]
    
    return res

# Useful method for running a time series test on a factor model
# Returns alphas and beta values
# df - the portfolio excess returns data
# factor_df - the factor excess returns data
# factors - a list of factors we're testing (ex: AQR = ['MKT', 'HML', 'RMW', 'UMD'])
def time_series_test_2(df, factor_df, factors, annualization=12):
    columns = ['alpha']
    for factor in factors:
        columns.append(f'{factor} Beta')
        
    ts_test = pd.DataFrame(data = None, index = df.columns, columns = columns)

        
    for asset in ts_test.index:
        y = df[asset]
        X = sm.add_constant(factor_df[factors])

        reg = sm.OLS(y, X).fit().params
        ts_test.loc[asset] = [reg[0] * annualization, reg[1], reg[2]]
    
    return ts_test


# To calculate factor premia (based on factor excess returns data) annualized
(factor_df.mean() * 12).to_frame('Factor Premia')

# To calculate mean absolute error of your time series test (ts_test)
# Mean Absolute Error: mean of the absolute values of the alphas from your regression
print('MAE: ' + str(round(ts_test['alpha'].abs().mean(), 4)))
# Example from HW 5: AQR_test['AQR alpha'].abs().mean()

# To compute predicted premium, use the following
# NOTE: for this code to work, the columns in ts_test must be named the same as the factor_data i.e. MKT=MKT, CL1=CL1
(factor_data.mean() * 12 * ts_test[['MKT','CL1']]).sum(axis = 1).to_frame('Predicted Premium')
# To get largest, 
(factor_data.mean() * 12 * ts_test[['MKT','CL1']]).sum(axis = 1).to_frame('Predicted Premium').nlargest(1, 'Predicted Premium')


# For Cross-Sectional Tests, run another regression of the excess returns against the betas of the time-series test
# Example 1:
# The dependent variable, (y): mean excess returns from each of the n = 25 portfolios.
y = portfolioData_excessReturns.mean()

# The regressor, (x): the market beta from each of the n = 25 time-series regressions.
X = sm.add_constant(capm_report['beta'])

res = sm.OLS(y,X,missing='drop').fit()
res.summary()
# Example 2:
y = commodities_df.mean()
X = sm.add_constant(ts_test[['MKT','CL1']].astype(float))

cross_sect = sm.OLS(y, X).fit()

# Mean Absolute Error:
#    Time Series Test:
df['alpha'].abs().mean()
#    Cross Sectional Test:
cross_sect.resid.abs().mean() * 12

# Predicted Premium:
#    Time Series Test: (returns (annualized) * betas).sum()
(factor_data.mean() * 12 * ts_test[['MKT','CL1']]).sum(axis = 1).to_frame('Predicted Premium')
#    Cross Sectional test: (alpha + (ts betas * cross sectional betas)).sum()
predicted = (cross_sect.params[0] + (ts_test[['MKT','CL1']] * cross_sect.params[1:]).sum(axis=1)) * 12

# Estimated Premia:
# In cross-sectional, it's the betas (see Midterm 2020 Q4)

# Cross Sectional Test
# Generally, put the excess returns on the left (y), and the time series betas on the right, with sm.add_constant(betas)
# Generates the betas from the regression on the factors
def ts_betas(df, factor_df, factors, intercept=False):
    if intercept == True:
        res = pd.DataFrame(data = None, index = df.columns, columns = ['alpha'])
        res[factors] = None
    else:
        res = pd.DataFrame(data = None, index = df.columns, columns = factors)
    
    for port in df.columns:
        y = df[port]
        if intercept == True:
            X = sm.add_constant(factor_df[factors])
        else:
            X = factor_df[factors]
        model = sm.OLS(y, X).fit()
        res.loc[port] = model.params
    
    return res

# This method runs a cross-sectional test for a portfolio given portfolio excess returns, factor excess returns, and the factors you want to include
def cross_section(df, factor_df, factors, ts_int=True, annualization=12):
    betas = ts_betas(df, factor_df, factors, intercept=ts_int)
    res = pd.DataFrame(data = None, index = betas.index, columns = factors)
    res['Predicted'] = None
    res['Actual'] = None
    
    for port in res.index:
        res.loc[port, factors] = betas.loc[port]
        prem = (betas.loc[port] * factor_df[factors]).sum(axis=1).mean() * annualization
        res.loc[port,['Predicted','Actual']] = prem, df[port].mean() * annualization
    
    return res

def cross_premia(df_cs, factors):
    y = df_cs['Actual'].astype(float)
    X = df_cs[factors].astype(float)

    return sm.OLS(y,X).fit().params.to_frame('CS Premia')

def cross_premia_mae(df_cs, factors, model):
    y = df_cs['Actual'].astype(float)
    X = df_cs[factors].astype(float)

    print(model + ' MAE: ' + str(round(sm.OLS(y,X).fit().resid.abs().mean(), 4)))


# Notes
'''
Fama-French 3/5 Factors:
    - Market: market factor    
    - Size Factor: Long the small stocks, short the big stocks
    - Value Factor: Long the value stocks (high book to market ratio), short the growth stocks (low book to market)
        Though the value factor generally had lower returns in the DFA case, it's low correlation made it a good diversifier
        Not all investors can be exposed to the value factor, otherwise it would be reflected in the market premium
    - Profitability Factor: Long the profitable (high operating profit to book value) stocks, short the unprofitable (low operating profit to book ratio)
    - Investment: Long the stocks with conservative (low) investment, short the stocks with aggresive investment, investment = percent change of asset value
    
If Fama-French were perfect:
    - It would span the tangency, i.e. portfolio has the highest Sharpe
    
    
Momentum Factor:
    - You need a large universe, with extreme winners and losers
    - Play both sides for diversification and hedging
    - long the stocks with large price appreciation, short the stocks with price depreciation
'''

'''
CAPM: a linear factor pricing model that tries to explain all expected returns of an asset with the expected returns of a portfolio constructed with
    every asset in the market (not SPY!!)
    
If CAPM were perfect:
    - the market portfolio = tangency portfolio
    - all returns explained by portfolio of market assets (ie in betas)
    - Treynor Ratio = market returns (same for all portfolios in HW 4 example)
    - Info Ratios = 0
    - We could run OLS and have alpha = 0
    - After running OLS R-squared is irrelevant
    - The intercept of time-series regressions is zero and that the intercept of the cross-sectional regression is zero.
    
Time-Series Test: used to test the model on assets (1st level of regression)
    - Visually, we looked at a graph of beta vs. excess return to validate if assets fall on risk-free rate/market line
    - If the model were perfect, we would expect alpha = 0 and a very low Mean Absolute Error

Mean Absolute Error: After running first regression, take abs(alpha).mean()

Cross-Sectional Test: OLS run on beta of Time Series Test to see if we can get it to fit a different line perfectly
    - This would happen if our estimate of the risk-free rate was off (i.e. a different intercept)
    
Mean Absolute Error (Cross-Sectional): Use the residuals of the regression- cross_sect.resid.abs().mean() * 12

Predicted Premium: 
    - For time series, take the annualized returns times the betas and add them all up
    - For cross-sectional predicted premium, take the time-series beta times the cross-sectional beta and sum them
            (and then either include or don't include intercept)

Adding a factor must improve the model- if F-F fails, CAPM must also fail
'''

'''
Midterm 2020 notes:
Why do we prefer to test Linear Factor Pricing Models on portfolios instead of on individual securities?
    - Because pricing tests are statistically very noisy. Individual securities have a lot of idiosyncratic risk, while 
      we want to concentrate on systematic risk. And working with portfolios gives us less idiosyncratic risk and higher statistical power.
      
Is the Momentum strategy robust to various construction methods? Explain.
    - Our investigation showed that the momentum strategy is fairly robust to choice of decile and sorting. All these constructions we 
    examined had similar statistical properties, though as we focus only on the top/bottom deciles we get higher mean return and higher volatility. 
    Note that momentum is NOT robust to including the short-side of the construction. Long-only is extremely correlated to the market return and is
    quite distinct from the long-short construction.
'''