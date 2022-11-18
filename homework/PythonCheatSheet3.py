import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from arch.univariate import GARCH, EWMAVariance 
from sklearn import linear_model
import scipy.stats as stats
from statsmodels.regression.rolling import RollingOLS
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 4)

# Used for regression and getting alpha, beta, and R-squared
def regress(y, X, intercept = True, annual_fac=12):
    if intercept == True:
        X_ = sm.add_constant(X)
        reg = sm.OLS(y, X_).fit()
        reg_df = reg.params.to_frame('Regression Parameters')
        reg_df.loc[r'$R^{2}$'] = reg.rsquared
        reg_df.loc['const'] *= annual_fac
    else:
        reg = sm.OLS(y, X).fit()
        reg_df = reg.params.to_frame('Regression Parameters')
        reg_df.loc[r'$R^{2}$'] = reg.rsquared
    
    return reg_df