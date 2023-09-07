#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:00:11 2023

@author: tristan
"""

import PSO
import pandas as pd
import glob
import numpy as np
from scipy.optimize import minimize, LinearConstraint




def sharpe_ratio(w,args):
    """
    Sharpe ratio computed for a portfolio made of the assets in quantities given by w

    Parameters
    ----------
    w : numpy array
        weight of the assets of the portfolio
    args : dict
        Additional parameters to compute the sharpe ratio:
            mu_l: average fo the daily returns of each asset
            cov: covariance matrix of the daily returns of the assets
            Rf: the risk-free rate
            T: the time horizon

    Returns
    -------
    float
        The sharpe ratio

    """
    mu_l=args["mu_l"]
    cov=args["cov"]
    Rf=args["Rf"]
    T=args["T"]
    return_p=mu_l.dot(w)
    covariance_p=cov.dot(w).dot(w)
    return (return_p-Rf)/np.sqrt(covariance_p/T)

def sharpe_ratio_minimise(w,args):
    # compute the inverse of the sharpe ratio, needed to use the scipy based minimise function
    return -sharpe_ratio(w, args)

def mu(M):
    #Average of the daily returns
    return np.mean(M,axis=0)

def returns(M):
    # Compute the returns from the adjusted close data
    R=M[1:,:]/M[:-1,:]-1
    return R

def covariance(M):
    #Covariance of the daily returns
    return np.cov(M.T)

def exponential_covariance(M):
    #Covariance of the daily returns computed using the exponential moving average
    N_assets=M.shape[1]
    cov=np.zeros((N_assets,N_assets))
    
    mu_l=mu(M)
    
    for i in range(N_assets):
        for j in range(i,N_assets):
            temp=0
            for k in range(len(M)):
                temp+= (M[-k,i]-mu_l[i])*(M[-k,j]-mu_l[j])*0.94**k
            cov[i,j]=temp*(1-0.94)
            cov[j,i]=cov[i,j]
    return cov




"""
Load the financial data
"""

folder="FTSE_data/"

component_l=glob.glob(folder+"*.csv")
df=pd.read_csv("^FTSE.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close'])

for c in component_l[:]:
    df_t=pd.read_csv(c,index_col="Date",parse_dates=True,usecols=['Date','Adj Close'])
    df_t=df_t.rename(columns={"Adj Close":f"Adj Close {c}"})
    df=df.merge(df_t,on="Date",how="left")

df=df.dropna()

"""
Compute the daily returns, their average and their covariance
"""
data=returns(df.to_numpy()[:,1:])
mu_l=mu(data)
cov=covariance(data)
# cov=exponential_covariance(data) #Using the exponential moving average formula

"""
Define the parameters required to compute the Sharpe ratio
Use T=1 for a time horizon of 1 day
T=63 for three months
"""

args={"mu_l":mu_l,"cov":cov,"Rf":0.0, "T":63}

# there are as many dimension to the problem as there are assets available
dimension=len(data[0])

# Bounds of the weight for each variable (needed for scipy.optimize.minimize function)
bounds=[]
for i in range(dimension):
    bounds.append((0,1))
    
#Constraint that the sum of the weights is equal to one (needed for scipy.optimize.minimize function)
linear_constraint = LinearConstraint([np.ones(dimension)], [1], [1])

"""
Optimisation of the portfolio by minimizing (- Sharpe ratio) using the scipy optimization function
"""
res=minimize(sharpe_ratio_minimise,np.ones(dimension)/dimension,args=args,bounds=bounds,constraints=[linear_constraint])
print(res.x,sharpe_ratio(res.x,args=args), res.x.sum())

 
"""
Optimisation of the portfolio by maximizing the Sharpe ratio using the particle swarm algorithm
"""
pso = PSO.ParticleSwarmOptimizer(iterations=10,swarm_size=1000,dimension=dimension)
solution=pso.optimize(sharpe_ratio,args=args,verbose=True)
print(solution,sharpe_ratio(solution, args=args),solution.sum())


