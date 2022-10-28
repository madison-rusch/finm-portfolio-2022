import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame()

# To plot the cumulative returns of a dataset of excess returns
((df + 1).cumprod() - 1).plot(title = "Full-sample")

# transpose(): used to switch the rows and the columns of a dataframe
df.transpose()



# Useful method for running a time series test on a factor model
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