import numpy as np
import pandas as pd
import cvxopt as opt
import streamlit as st
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import json
pd.options.display.float_format = '{:.2%}'.format

class UOptimizer: 
    def __init__(self, path, include_unison=True, beta=0.4, alpha=0.01):
        """ does what it says """
        self.path = path
        self.data = self.load_data(path)
        self.include_unison=include_unison
        self.beta=beta
        self.alpha=alpha
        self.weights = None
        
        self.return_to_use = None
        self.annualized_return_to_use = None
        self.annualized_vol_to_use = None
        #set returns and volatility
        self.set_returns_vols(12)
       
    def set_unison_alpha_beta(alpha, beta):
        """ does what it says """
        self.alpha = alpha
        self.beta = beta
        
    def set_returns_vols(self, period): 
        """ does what it says """
        self.return_to_use = self.logReturns(self.data, period).dropna()
        if self.include_unison:
            self.return_to_use["Unison"] = self.beta * (self.return_to_use['CSUSHPINSA'] + self.alpha)
        self.annualized_vol_to_use = self.return_to_use.std() * np.sqrt(12/period)
        self.annualized_return_to_use = self.return_to_use * (12/period)
        self.cleanup()
            
    @staticmethod
    def load_data(path):
        """ does what it says """
        data = pd.read_excel(path, sheet_name='Sheet1').iloc[2:]
        data.columns = data.iloc[0]
        data = data.iloc[1:].reset_index().drop(columns = ['index'])
        data.set_index('Date', inplace=True)

        #keep the indices which have atleast 60% data!
        data.dropna(thresh=len(data)*0.9, axis=1, inplace=True)
        data = data.astype('float')

        #interpolate the missing values, sort using index, take the log!
        data = data.interpolate()
        data = data.sort_index()
        data = np.log(data)
        return data
      
    @staticmethod
    def logReturns(df, period):
        """ does what it says """
        logRet = df.diff(period)
        return logRet.dropna()

    def cleanup(self):
        """ does what it says """
        stocks_to_be_dropped = ['LUCRTRUU Index','LGL1TRUU Index', 'MLCUWXU Index']
        self.return_to_use = self.return_to_use.drop(stocks_to_be_dropped, axis=1)
        self.annualized_vol_to_use = self.annualized_vol_to_use.drop(stocks_to_be_dropped)
        self.annualized_return_to_use = self.annualized_return_to_use.drop(stocks_to_be_dropped, axis=1)
        
    @staticmethod
    def optimal_portfolio(returns, low_weight_bound, high_weight_bound):
        """ does what it says """
        # Turn off progress printing
        solvers.options['show_progress'] = False
        returns = np.asmatrix(returns.T)                # -> (n_assets, n_observations)
        n_assets = len(returns)

        # Vector of desired returns
        N = n_assets*(int(1e+2))
        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

        # Obtain expected returns and covariance
        m1 = np.mean(returns, axis=1)                       # Mean returns
        c1 = np.cov(returns, bias=True)                     # Volatility (in terms of standard deviation)
        # Convert to cvxopt matrices
        pbar = opt.matrix(m1)
        S = opt.matrix(c1)

        # Create constraint matrices
        G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
        h = opt.matrix(np.vstack((low_weight_bound, high_weight_bound)))
        A = opt.matrix(1.0, (1, n_assets))
        b = opt.matrix(1.0)
        st.write(np.vstack((-np.eye(n_assets), np.eye(n_assets))).shape)
        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]
        sol = solvers.qp(S, -pbar, G, h, A, b)

        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        weights = [np.asarray(x) for x in portfolios]
        returns = np.asarray([blas.dot(pbar, x) for x in portfolios])
        risks = np.asarray([np.sqrt(blas.dot(x, S * x)) for x in portfolios])
        sharpe = returns/risks
        max_sharpe_idx = np.argmax(sharpe)
        min_vol_idx = np.argmin(risks)
        
        #UOptimizer.matplot_eff_frontier(returns, risks, sharpe)
        return weights, np.asarray(returns), np.asarray(risks), sharpe
    
    @staticmethod
    def matplot_eff_frontier(returns, risks, sharpe):
        """matplot lib version of efficient frontier"""
        ax_sharpe_idx = np.argmax(sharpe)
        min_vol_idx = np.argmin(risks)

        max_sharpe_idx = np.argmax(sharpe)
        min_vol_idx = np.argmin(risks)
        
        # Plot Efficient Frontier
        fig, ax = plt.subplots()
        plt.plot(risks, returns, 'y-o')
        plt.plot(risks[max_sharpe_idx], returns[max_sharpe_idx], '*', label = 'max_sharpe')
        plt.plot(risks[min_vol_idx], returns[min_vol_idx], '*', label = 'min_vol')
        plt.title('Efficient Frontier')
        plt.ylabel('Mean of Portfolio Returns')
        plt.xlabel('Standard Deviation of Portfolio Returns')
        plt.grid()
        plt.legend()
        st.pyplot(fig)
        
    @staticmethod
    def plotly_eff_frontier():
        """ does what it says """
        pass
    
    def parse_weights(self, weights_dict):
        """ does what it says """
        low_weight_bound = [i[0]/100 for k, i in weights_dict.items() if self.include_unison or k!="Unison"]
        high_weight_bound = [i[1]/100 for k, i in weights_dict.items() if self.include_unison or k!="Unison"]
        return np.asarray(low_weight_bound).reshape(-1,1), np.asarray(high_weight_bound).reshape(-1,1)
    
    def summarize(self, returns, risks, sharpe):
        weights = self.weights
        ind_opt = np.argmax(sharpe)            # Index of selected portfolio

        opt_portfolio = {}
        opt_portfolio['return'] = returns[ind_opt] 
        opt_portfolio['risk'] = risks[ind_opt] 
        opt_portfolio['sharpe'] = sharpe[ind_opt]

        wt = weights[ind_opt]/sum(weights[ind_opt])
        ind_w = np.flip(np.argsort(wt, axis=0), axis=0)
        opt_portfolio['weights'] = wt[ind_w]
        ind_w = ind_w.ravel().tolist()
        sym1 = pd.DataFrame(list(ret))

        sym=sym1.loc[ind_w]

        #sym = [str(sym[k][0][0]) for k in range(len(sym))]
        opt_portfolio['stocks'] = sym

        output = pd.DataFrame(columns=["Ticker","Weights%"])
        output["Ticker"] = sym[0]
        output["Weights%"] = wt[ind_w]
        output = output.reset_index(drop=True)
        st.write(output)
        
    def optimize_main(self, weights_dict):
        """ does what it says """
        ret = self.return_to_use
        low_weight_bound, high_weight_bound = self.parse_weights(weights_dict)
        weights, returns, risks, sharpe = self.optimal_portfolio(ret, 
                                                                 low_weight_bound, 
                                                                 high_weight_bound)
        
        self.weights = weights
        return returns, risks, sharpe