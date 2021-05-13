import streamlit
import os
import glob
import time
import multiprocessing
import logging

import streamlit as st
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2%}'.format

import plotly.graph_objects as go
from optimizer import UOptimizer
#from Inputs_Parallel import get_possible_scenarios
import chart_studio.plotly as py

import matplotlib.pyplot as plt

def plotly_eff_frontier(optimizer, optimizer_unison, weights):
    """ does what it says """
    # Graphing Function #####
    returns, risks, sharpe = optimizer.optimize_main(weights)
    fig = go.Figure(data=[go.Scatter(x=risks,
                            y=returns,
                            #hoveron=sharpe,
                            mode= "lines", #'lines+markers', #"lines"   
                            name = 'Without Unison',
                            #line=go.scatter.Line(color="gray"),
                            #showlegend=False)
                            marker=dict(
                                    size=10,
                                    color=sharpe, #set color equal to a variable
                                    colorscale='Viridis', # one of plotly colorscales
                                    showscale=True
                                )
                            )
                         ])

    returns1, risks1, sharpe1 = optimizer_unison.optimize_main(weights)
    fig.add_trace(
        go.Scatter(
            x=risks1,
            y=returns1,
            mode="lines",#"markers",
            name="With Unison",
            marker=dict(
                        size=3,
                        color=sharpe, #set color equal to a variable
                        #colorscale='Viridis', # one of plotly colorscales
                        showscale=True
                    )
            #line=dict(color="black")
        )
    )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    fig['layout']['yaxis'].update(autorange=True, rangemode='tozero')
    fig['layout']['xaxis'].update(autorange=True, rangemode='tozero')
    fig.update_layout(hovermode='x unified')
    fig.update_layout(title='Efficient Frontier', autosize=True,
                      xaxis=dict(
                                title=dict(
                                  text='Standard Deviation of Portfolio Returns'
                            )),
                      yaxis=dict(
                                title=dict(
                                  text='Mean of Portfolio Returns'
                            )),
                      #width=800, height=800,
                      margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)
    
    st.subheader('Without Unison: Max-Sharpe Weights')
    #optimizer.matplot_eff_frontier(x, y, sharpe)
    optimizer.summarize(returns, risks, sharpe)
    
    st.subheader('With Unison: Max-Sharpe Weights')
    #optimizer_unison.matplot_eff_frontier(x1, y1, sharpe1)
    optimizer_unison.summarize(returns, risks, sharpe1)
    

def main(optimizer, optimizer_unison):
    # Side Bar #######################################################
    #think about using 'collapsible_container' in fuutre

    st.sidebar.subheader('Current Portfolio Weights')
    pf_wt = {}
    for c in optimizer.annualized_return_to_use.columns:
        if c != "CSUSHPINSA": 
            pf_wt[c] = st.sidebar.number_input(label=optimizer.get_name_dict().get(c, "") + " " + c, 
                                               value=0.1)
            
    st.sidebar.subheader('Asset Weight Limits For Optimization')
    weights = {}
    
    for c in optimizer.annualized_return_to_use.columns:
        if c != "CSUSHPINSA": # don't use case-shiller here
            weights[c] = st.sidebar.slider(label=optimizer.get_name_dict().get(c, "") + " " + c,
                                          min_value=0,
                                          max_value=100,
                                          step=1,
                                          value=(0, 30))
    
    st.sidebar.subheader('Borrow Limit')
    a = st.sidebar.slider(label="Borrow Asset",
                                          min_value=0,
                                          max_value=100,
                                          step=1,
                                          value=(0, 30))
    
    st.sidebar.subheader('Unison Proxy (Case–Shiller) Allocation')
    weights['Unison'] = st.sidebar.slider(label="Proxy (Case–Shiller)",
                                          min_value=0,
                                          max_value=100,
                                          step=1,
                                          value=(0, 30))
    
    #run_button = st.sidebar.button(label='Run Optimization')
    
    #in case we want to give user the flexibility to 
    unison_beta = st.sidebar.number_input("Unison Beta", value=2.1)
    unison_alpha = st.sidebar.number_input("Unison Alpha", value=0.01)
    optimizer_unison.set_unison_alpha_beta(unison_alpha, unison_beta)
    
    # App ##################################################
    st.title("Unison Portfolio Optimization")
    plotly_eff_frontier(optimizer, optimizer_unison, weights)
    
    
#@st.cache
def load_optimizer():
    optimizer = UOptimizer('timeseriesUpdated.xlsx', include_unison=False)
    optimizer_unison = UOptimizer('timeseriesUpdated.xlsx')
    return (optimizer, optimizer_unison)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    optimizer, optimizer_unison = load_optimizer()
    main(optimizer, optimizer_unison)
