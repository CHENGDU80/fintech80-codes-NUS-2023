# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:21:08 2023

@author: KJ
"""

import streamlit as st
import pandas as pd
import numpy as np
# import requests
# from selenium import webdriver
# import pandas as pd
# from bs4 import BeautifulSoup
import time
import plotly.express as px
# import yfinance as yf
from datetime import datetime
# import networkx as nx
import matplotlib.pyplot as plt
# import json
# import streamlit.components.v1 as components



# Set Streamlit page configuration to a wide layout
st.set_page_config(layout="wide")
# Define a path for reading data
debug = 'C:/users/kj/Desktop/cd-action/data/'


# Title and introductory text for the home page

# Header and subheader for the "Company Specific Analysis" section
st.title('Actionable Suggestion for Apple Inc. (AAPL)')
st.header('1. News Analysis Report')
st.text('Generating report... Done!')
st.write('We are summarising your final equity report. By combining our analysis from Macro markets, Industrial news and company details, this report will give you the comprehensive top-down analysis from marco to company news overview, and the unbiased impact analysis to the company brought by the news events.')

with open(debug+"dummy.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Export News Analysis Report",
                    data=PDFbyte,
                    file_name="News_Analysis_Report.pdf",
                    mime='application/octet-stream')


st.header('2. Risk Index and Influences')
cola, colb = st.columns(2)

with cola:
    st.metric(label="Overall Risk Score", value="0.26", delta="0.05 w-o-w", delta_color='inverse')
with colb:
    risk_ts = pd.read_csv(debug+'Risk_timeseries.csv')
    risk_ts['Date'] = pd.to_datetime(risk_ts['Date'], format='%Y/%m/%d')
    fig1 = px.line(risk_ts, x='Date', y='Index', labels={'Value': 'Risk Index'}, title='Risk Index over Time')
    fig1.update_traces(line_color='#6B5B95')
    # fig1.update_layout(template="plotly_white")
    # fig1.write_html("data/risk_timeseries.html")
    st.plotly_chart(fig1, use_container_width=True)

risk_bar = pd.read_csv(debug+'Risk_bar.csv')
risk_bar['Absolute_Value'] = risk_bar['Score'].abs()
risk_bar = risk_bar.sort_values(by='Absolute_Value', ascending=True)

fig2 = px.bar(risk_bar, x='Score', y='Indicator', text='Score', color='Score', 
    color_continuous_scale=[(0, '#DD4124'), (1, '#009B77')], title='Most Important Financial Risk Indicators')
# fig2.update_layout(template="plotly_white")
# fig2.write_html('data/risk_barplot.html')
st.plotly_chart(fig2, use_container_width=True)

