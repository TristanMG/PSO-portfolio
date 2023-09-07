Implementation of a particle swarm solver to optimise a portfolio by maximising its Sharpe ratio.

Financial data is downloaded from yahoo finance using the file download_data.py. Only 30 components of FTSE are considered and are stored in the folder FTSE_data.

The particle swarm solver is coded in PSO.py and works for any maximisation problem with the constraint that every variable of the problem is boundded by 0 and 1 and that the sum of the variables is equal to 1.

The implementation of the solver to find an optimal portfio by maximising the Sharpe ratio,  and a comparison with a direct method from scipy.optimize, is in portfolio_optimisation.py  
