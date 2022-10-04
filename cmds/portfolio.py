import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns





def tangency_weights(returns,dropna=True,scale_cov=1):
    if dropna:
        returns = returns.dropna()

    covmat_full = returns.cov()
    covmat_diag = np.diag(np.diag(covmat_full))
    covmat = scale_cov * covmat_full + (1-scale_cov) * covmat_diag

    weights = np.linalg.solve(covmat,returns.mean())
    weights = weights / weights.sum()

    return pd.DataFrame(weights, index=returns.columns)


        


def MVweights(returns=None, mean=None, cov=None, target=None, isexcess=True):
    if returns is not None:
        rets = returns.copy()
        rets.dropna(inplace=True)

        mean = rets.mean()
        cov = rets.cov()
        
    wtsTan = np.linalg.solve(cov,mean)
    wtsTan /= wtsTan.sum()
    muTan = np.inner(wtsTan,mean)
    
    wtsGMV = np.linalg.solve(cov,np.ones(mean.shape))
    wtsGMV /= wtsGMV.sum()
    muGMV = np.inner(wtsGMV,mean)
    
    if target == 'GMV':
        target = muGMV
        isexcess = False
    elif target == 'TAN':
        target = muTan
    elif target is None:
        target = muTan
    
    if isexcess:
        share_tangency = target/muTan
        wtsGMV = 0
    else:
        share_tangency = (target - muGMV)/(muTan-muGMV)
    
    wstar = share_tangency * wtsTan + (1-share_tangency) * wtsGMV
    
    return wstar




def performanceMetrics(returns,annualization=1, quantile=.05):
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Mean'] = returns.mean() * annualization
    metrics['Vol'] = returns.std() * np.sqrt(annualization)
    metrics['Sharpe'] = (returns.mean() / returns.std()) * np.sqrt(annualization)

    metrics['Min'] = returns.min()
    metrics['Max'] = returns.max()
    return metrics
        
