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

def main(optimizer, optimizer_unison):
    # Side Bar #######################################################
    st.sidebar.subheader('Asset Weights')

    weights = {}
    for c in optimizer.annualized_return_to_use.columns:
        weights[c] = st.sidebar.slider(label=c,
                                          min_value=0,
                                          max_value=100,
                                          step=1,
                                          value=(0, 30))
    st.sidebar.subheader('Unison Allocation')
    
    weights['Unison'] = st.sidebar.slider(label="Unison",
                                          min_value=0,
                                          max_value=100,
                                          step=1,
                                          value=(0, 30))
    run_button = st.sidebar.button(label='Run Optimization')
    unison_beta = st.sidebar.number_input("Unison Beta", value=0.4)
    unison_alpha = st.sidebar.number_input("Unison Alpha", value=0.01)
    
    # App ###########################################################
    st.title("Unison Portfolio Optimization")

    # Graphing Function #####
    x, y, sharpe = optimizer.optimize_main(weights)
    fig = go.Figure(data=[go.Scatter(x=y,
                            y=x,
                            #hoveron=sharpe,
                            mode='lines+markers', #"lines"   
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
    
    x1, y1, sharpe1 = optimizer_unison.optimize_main(weights)
    fig.add_trace(
        go.Scatter(
            x=y1,
            y=x1,
            mode="markers",
            name="With Unison",
            marker=dict(
                        size=3,
                        color=sharpe, #set color equal to a variable
                        #colorscale='Viridis', # one of plotly colorscales
                        showscale=True
                    )
            
            #line=dict(
            #    color="black"
            #)
        )
    )
    
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    
    fig.update_layout(hovermode='x unified')

    fig.update_layout(title='Efficient Frontier', autosize=True,
                      xaxis=dict(
                                title=dict(
                                  text='Standard Deviation of Portfolio Returns'
                                )
                              ),
                      yaxis=dict(
                                title=dict(
                                  text='Mean of Portfolio Returns'
                                )
                              ),
                      #width=800, height=800,
                      margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)
    
    optimizer.matplot_eff_frontier(x, y, sharpe)
    optimizer.summarize(x, y, sharpe)
    
    optimizer_unison.matplot_eff_frontier(x1, y1, sharpe1)
    optimizer_unison.summarize(x1, y1, sharpe1)
    
#@st.cache
def load_optimizer():
    optimizer = UOptimizer('timeseriesUpdated.xlsx', include_unison=False)
    optimizer_unison = UOptimizer('timeseriesUpdated.xlsx')
    return (optimizer, optimizer_unison)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    optimizer, optimizer_unison = load_optimizer()
    main(optimizer, optimizer_unison)
