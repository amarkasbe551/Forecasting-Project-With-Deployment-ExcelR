
import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import datetime

#################################################################################
# Title
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('https://cdn.discordapp.com/attachments/1029411657090334753/1066985432694394930/PngItem_52517.png', width=100)
with col2:
    st.title("Stock Price Predictor")


# import the dataset
user_input = st.text_input('Enter Stock Ticker','RELIANCE.NS')
df = yf.download(user_input ,start = '2015-1-1').round(4)
#coping data set
df1 = df.copy()
#resetting index
df = df.reset_index()

#Data Set
st.subheader('Data Set')
st.write(df)

#Statistics Summary of Data
st.subheader('Statistics Summary of a dataframe.')
st.write(df.describe())


# Candlestick
st.subheader('Candlestick Chart with 50 EMA and 100 EMA')

# add moving averages to df
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA100'] = df['Close'].rolling(window=100).mean()

fig = go.Figure()
# add OHLC trace
fig.update_layout(xaxis_title="Date",
                  yaxis_title="Price",
                  width=1000,
                  height=800)
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             showlegend=False))

# add moving average traces
fig.add_trace(go.Scatter(x=df.index,
                         y=df['MA50'],
                         opacity=0.7,
                         line=dict(color='pink', width=2),
                         name='MA 50'))

fig.add_trace(go.Scatter(x=df.index,
                         y=df['MA100'],
                         opacity=0.7,
                         line=dict(color='cyan', width=2),
                         name='MA 100'))
# removing white space
fig.update_layout(margin=go.layout.Margin(
        l=20, #left margin
        r=20, #right margin
        b=20, #bottom margin
        t=20  #top margin
    ))
# Adding rangeslider
fig.update_layout(xaxis_rangeslider_visible=True)
# add chart title
st.plotly_chart(fig)


####################### MODEL BUILDING ###############################

# Making the Date as DateTime index for the Dataframe.
df.set_index('Date',inplace=True)
# Resampling day wise
upsampled_day = df.resample('D').mean()
# Interpolation was done for nan values which we get after doing upsampling by day
df_day = upsampled_day.interpolate(method='linear')


# Building Final Model And Ploting Actual vs Predicted For Daily Data
triple_exp = ExponentialSmoothing(df_day["Close"], trend = 'add', seasonal = "add", seasonal_periods=10).fit()
df_day["pred_triple_exp"] = triple_exp.predict(start = df_day.index[0],end = df_day.index[-1])

# Visualizing Actual VS Predict
st.subheader('Actual VS Predict')
fig1 = go.Figure()
fig1.update_layout(xaxis_title="Date",
                  yaxis_title="Price",
                  width=1000,
                  height=800)
fig1.add_trace(go.Scatter(x=df.index,
                         y=df_day["Close"],
                         opacity=0.7,
                         line=dict(color='red', width=3),
                         name='Actual'))
fig1.add_trace(go.Scatter(x=df.index,
                         y=df_day["pred_triple_exp"],
                         opacity=0.7,
                         line=dict(color='cyan', width=1),
                         name='Predict'))
# removing white space
fig1.update_layout(margin=go.layout.Margin(
        l=20, #left margin
        r=20, #right margin
        b=20, #bottom margin
        t=20  #top margin
    ))
# Adding rangeslider
fig1.update_layout(xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

################################ Forecasting Model #########################################
# Forecasting for next 30 days

st.subheader('Forecasting For Next 30 Days')
triple_exp = triple_exp.forecast(30)
fig3 = px.line(triple_exp)
fig3.update_layout(xaxis_title="Date",
                  yaxis_title="Price",
                  width=950,
                  height=800,showlegend=False,xaxis_rangeslider_visible=True)
fig3.update_traces(line_color="#32CD32")
st.plotly_chart(fig3)



















