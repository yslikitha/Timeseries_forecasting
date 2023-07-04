import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import csv

# load the data
data = pd.read_csv('traffic_volume2.csv')
data1 = pd.read_csv('traffic_volume4.csv')

dates = pd.date_range(start='1/1/2018', periods=len(data), freq='MS')


# create a SimpleExpSmoothing object with alpha value 0.3 and fit the model
model = SimpleExpSmoothing(data['volume'])
model1 = SimpleExpSmoothing(data1['volume'])

fit_model = model.fit(smoothing_level=0.3)
fit_model1 = model1.fit(smoothing_level=0.3)

# generate smoothed values for the historical data
smoothed_values = fit_model.fittedvalues
smoothed_values1 = fit_model1.fittedvalues

# generate forecasts for the next 4 months
forecast = fit_model.forecast(4)
forecast_dates = pd.date_range(start='1/1/2021', periods=4, freq='MS')
forecast1 = fit_model1.forecast(4)
forecast_dates1 = pd.date_range(start='1/1/2019', periods=4, freq='MS')

# calculate mean squared error
mse = mean_squared_error(data['volume'].iloc[12:16], forecast1)
rmse = math.sqrt(mse)
print("Root Mean Square Error: ", rmse)

# plot the historical data, smoothed values, and forecasted values
plt.plot(dates, data['volume'], label='historical data')
plt.plot(dates, smoothed_values, label='smoothed values')
plt.plot(forecast_dates, forecast, label='forecasted values')
plt.plot(forecast_dates1, forecast1, label='forecasted values_testing')
plt.title('Traffic Volume Forecast using Simple Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()
