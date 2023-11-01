# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:21:08 2023

@author: KJ
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
import time
import plotly.express as px
import yfinance as yf
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import json
import streamlit.components.v1 as components



# Set Streamlit page configuration to a wide layout
st.set_page_config(layout="wide")
# Define a path for reading data
debug = 'C:/users/kj/Desktop/chengdu80-nus/data/'


# Title and introductory text for the home page

# Header and subheader for the "Company Specific Analysis" section
st.title('Company Specific Analysis')
st.header('Select the company you want to investigate')
# Dropdown selector for choosing a company
ticker = st.selectbox(label='', options=['Apple Inc. (AAPL)','Microsoft Corporation (MSFT)','NVIDIA Corporation (NVDA)'], placeholder='Select Stock...') 

 
if not (ticker is None):
    # Extract the stock ticker from the selected company
    string = ticker[-5:-1]
    # Read news data from a CSV file
    news = pd.read_csv(debug+string+'_news_yahoo_sent.csv')
    news['date'] = pd.to_datetime(news['date'], format="%Y-%m-%d")
    # news = pd.read_csv('data/'+string+'_news_yahoo.csv')
    news = news.drop(news.columns[0], axis=1).loc[:,['date','sentiment','title','desc','source','url']]

    st.divider()
    st.header('Company News')
    # Display the recent company news in an expander
    with st.expander("View all company news from past week"):
      # Function to color code sentiment in the dataframe
      def colorful_sentiment(value):
        return f"background-color: pink;" if value in ["negative"] else f"background-color: lightgreen;" if value in ["positive"] else None
      st.dataframe(news.style.applymap(colorful_sentiment), 
                   #Column link to URL
                   column_config={"url": st.column_config.LinkColumn()},
                   use_container_width =True)
    st.text(str(news.shape[0])+' news found... Sentiments computed!')
    st.download_button("Download all recent company news",
    news.to_csv(),
    "file.csv",
    "text/csv",
    key='download-csv')

    cola, colx = st.columns(2)
    with cola:  
      #Company summary
      st.subheader("**Summary of key recent company news**")
      st.write((str(news.shape[0])+' news articles summarised with Prompt Engineering and LLM'))
    with colx:
      more = st.selectbox('Adjust number of words in summary', options=[50,200])
    # coy_summary = json.load(open(debug+string+'_summary.json'))
    # st.text(coy_summary['text'])
    coy_summary = pd.read_csv(debug+string+'_summary_dataframe.csv')
    if more==50:
      st.text(coy_summary.iloc[0]['0'])
    else:
      st.text(coy_summary.iloc[1]['0'])


    st.divider()
    st.header('Stock Price and Sentiments')
    compare = st.checkbox(label='Compare Price to S&P500')
    # # Create two columns for UI layout
    # colz, colb = st.columns(2)
    # # with cola:
    # #   # Download button to export the news data to a CSV file
    # #   st.download_button("Download all recent company news",
    # #   news.to_csv(),
    # #   "file.csv",
    # #   "text/csv",
    # #   key='download-csv')
    # with colb:
    #   # Checkbox to choose whether to compare with S&P500
    #   compare = st.checkbox(label='Compare to S&P500')


    # Create two columns for plotting data
    col1, col2 = st.columns(2)
    with col1:
      # # importing the yfinance package
      # import yfinance as yf

      # # giving the start and end dates
      # startDate = '2022-10-29'
      # endDate = datetime.now().strftime(format='%Y-%m-%d')

      # # downloading the data of the ticker value between
      # # the start and end dates
      # resultData = yf.download(string, startDate, endDate)
      resultData = pd.read_csv(debug+string+'_price.csv')
      resultData = resultData.set_index('Date')

      if compare:
        # # importing the yfinance package
        # import yfinance as yf

        # # giving the start and end dates
        # startDate = '2022-10-29'
        # endDate = datetime.now().strftime(format='%Y-%m-%d')

        # # downloading the data of the ticker value between
        # # the start and end dates
        # resultData2 = yf.download('%5EGSPC', startDate, endDate)
        resultData2 = pd.read_csv(debug+'S&P500_price.csv')
        resultData2 = resultData2.set_index('Date')

        # Normalize and compare stock and S&P500 data
        rebase = pd.concat([resultData.rename(columns={'Close':string}),resultData2.rename(columns={'Close':'S&P500'})], axis=1)
        rebase = (rebase / rebase.iloc[0]) * 100 -100
        # Create a plot for the comparison
        fig = px.line(rebase, x=rebase.index, y=[string, 'S&P500'], title=ticker+ ' against S&P500', 
                      labels={'value': 'Cumulative Return (%)'})
        st.plotly_chart(fig, use_container_width=True)
      
      else:
        # Create a plot for the stock price over time
        fig = px.line(resultData, x=resultData.index, y='Close', title=ticker+ ' Over Time',
                      labels={'Close': 'Stock Price (USD)'})
        st.plotly_chart(fig, use_container_width=True)
    

    with col2:
      # Define a function that maps sentiments to numerical values
      def mapsent(x):
          return 1 if x=='positive' else -1 if x=='negative' else 0 

      # Add a numerical sentiment column to the news dataframe
      news['sentiment_val'] = news['sentiment'].apply(mapsent)

      # Calculate and plot the sentiment trend over time
      senti_line = news.groupby('date')['sentiment_val'].agg(lambda x: x.sum()/x.count())
      fig2 = px.line(senti_line, x=senti_line.index, y='sentiment_val', title='News Sentiment of '+ticker, 
      labels={'date': 'Date', 'sentiment_val': 'Sentiment'})

      # Change the line color and add a horizontal dashed line at y=0
      fig2.update_traces(line_color='orange')
      fig2.add_shape(type='line', x0=senti_line.index.min(), x1=senti_line.index.max(),
          y0=0, y1=0, line=dict(color='black', width=1, dash='dash') )

      st.plotly_chart(fig2,use_container_width=True)


    st.divider()
    st.header("Multi-angle Analysis")
    st.text("""Computing news perspective clusters using K-means... Computed! \nSummarising news clusters with Prompt Engineering and LLM... Summarised!""")
    mult = pd.read_csv(debug+string+'_multi_angle.csv')
    # for i in range(mult.shape[0]):
    #   st.write(mult.iloc[i,1])
    tab_labels = mult["Unnamed: 0"].unique().tolist()
    i=0
    for tab in st.tabs(tab_labels):
        with tab:
            st.write(mult.iloc[i,1])
        i=i+1


    st.divider()
    st.header("News Impact Analysis")
    # Create two columns for plotting data
    col3, col4 = st.columns(2)

    with col3:
      st.markdown("What is the impact of recent **:red[Negative News]** 😡 on "+ticker+"?")
      # st.write('What is the impact of **Negative News** on '+ticker+"?")
      neg_impact = pd.read_json((debug+string+'_negative_impact.json'))
      neg_news = st.selectbox(label='', options=neg_impact['negative_title'], index=None, placeholder='Select Negative News...')
      if neg_news is not None:
        st.write(neg_impact[neg_impact['negative_title']==neg_news].iloc[0,2])

    with col4:
      st.markdown("What is the impact of recent **:green[Positive News]** 🤗 on "+ticker+"?")
      # st.write('What is the impact of **Negative News** on '+ticker+"?")
      pos_impact = pd.read_json((debug+string+'_positive_impact.json'))
      pos_news = st.selectbox(label='', options=pos_impact['positive_title'], index=None, placeholder='Select Positive News...')
      if pos_news is not None:
        st.write(pos_impact[pos_impact['positive_title']==pos_news].iloc[0,2])      




    st.divider()
    st.header("News Development Tracing")
    col5, col6 = st.columns(2)

    with col5:
      final_results_df = pd.read_csv(debug+string+"_trace.csv")
      final_results_df=final_results_df.drop(final_results_df.columns[0], axis=1)
      final_results_df['date'] = pd.to_datetime(final_results_df['date'])
      df = news.copy()
      df['date'] = pd.to_datetime(df['date'])

      date_range = pd.date_range(start=final_results_df['date'].min(), end=final_results_df['date'].max())
      # Group by 'date' and 'sentiment', and then count the occurrences
      sentiment_counts = final_results_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
      # Reindex the sentiment_counts to make sure every date in the date range is included
      # Unstack the sentiment to make sure we have all sentiments for each date
      complete_sentiment_counts = (
          sentiment_counts.set_index(['date', 'sentiment'])
          .reindex(pd.MultiIndex.from_product([date_range, df['sentiment'].unique()], names=['date', 'sentiment']), fill_value=0)
          .reset_index()
      )
      color_map = {
          'negative': 'red',
          'neutral': 'grey',
          'positive': 'green'
      }
      # Sort the sentiments by your desired order
      ordered_sentiments = ['negative', 'neutral', 'positive']
      complete_sentiment_counts['sentiment'] = pd.Categorical(complete_sentiment_counts['sentiment'], categories=ordered_sentiments, ordered=True)
      # Create a line plot with Plotly
      fig = px.line(
          complete_sentiment_counts,
          x='date',
          y='count',
          color='sentiment',
          title='Sentiment Counts Over Time For '+ticker,
          labels={'count': 'Number of Occurrences', 'date': 'Date'},
          color_discrete_map=color_map  # Use the color map for discrete colors based on sentiment
      )
      # Customize the markers for visibility and add different dash styles
      fig.update_traces(
          mode='lines+markers',
          marker=dict(size=8) 
      )
      # Apply different dash patterns for each sentiment
      # (assuming you want different dash styles to distinguish the lines)
      for i, sentiment in enumerate(ordered_sentiments):
          fig.for_each_trace(
              lambda t, pattern=i: t.update(line=dict(dash=['solid', 'dot', 'dash'][pattern])) if t.name == sentiment else (),
          )
      # Set y-axis to start from 0
      fig.update_yaxes(range=[0, complete_sentiment_counts['count'].max() + 1])
      # Show the plot
      st.plotly_chart(fig, use_container_width=True)

    with col6:
      title = final_results_df['title'].unique()[0]
      desc = final_results_df['original_desc'].unique()[0]
      st.subheader('**Find out how events in this news develop over time**')
      st.write("**Headline:** "+title + "\n\n" + "**Preview:** "+desc)




    st.divider()
    st.header("**Value Chain Analysis**")
    # st.write("""Firms frequently mentioned alongside """+ticker+""" in news articles are extracted and plotted 
    # to show the tight-knit relationship with """+ticker+""". Large nodes indicate a higher co-occurance in news articles""")
    st.text("""Extracting companies using Named Entity Recognition... Extracted! \nPlotting network of co-occuerence... Plotted!""")
    st.write("""These companies frequently co-occur in news articles with **""" + ticker + """**. Larger nodes indicate higher co-occurence""")
    st.markdown('''- **:red[Red Nodes]**: These companies are appear in **:red[Negative Sentiment News]** together with ''' + ticker)
    st.markdown("- **:green[Green Nodes]**: These companies are appear in **:green[Positive Sentiment News]** together with " + ticker)

    # HtmlFile = open(debug+"net.html", 'r', encoding='utf-8')
    # source_code = HtmlFile.read() 
    # components.html(source_code)

    p = open(debug+string+"_net.html")
    components.html(p.read(), height=500, width=750)

    # st.write(debug+"net.html", unsafe_allow_html=True)
    # #Read graph data
    # graph = pd.read_csv(debug+string+'_links_fix.csv')
    # graph = graph.drop(graph.columns[0], axis=1).groupby('Key').sum('Value').reset_index()
    # if string=='MSFT':
    #   graph = graph[graph['Value']>=3]
    # else:
    #   graph = graph[graph['Value']>=5]
    # graph['source'] = string
    # graph['target'] = graph['Key']
    # graph['weight'] = graph['Value']

    # #Create the graph and node weights
    # Graphtype = nx.Graph()
    # G = nx.from_pandas_edgelist(graph, edge_attr='weight',create_using=Graphtype)
    # weighted_degrees = dict(G.degree(weight='weight'))

    # # Step 3: Plot the graph with node sizes proportional to their weighted degrees
    # node_sizes = [15 * weighted_degrees[node] for node in G.nodes()]
    # node_sizes[0] = min(node_sizes)

    # net = plt.figure()
    # nx.draw_networkx(G, node_size=node_sizes)
    # st.pyplot(net) 



 