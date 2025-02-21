#!/usr/bin/env python
# coding: utf-8

# # Data Pre - Proccessing

# In[95]:


import pandas as pd
data = pd.read_csv("Electricity Demand Forecast.csv")


# In[2]:


data


# In[3]:


data.isnull().sum()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data["solar_exposure"].unique()


# In[7]:


import numpy as np
data.replace('Nodata', np.nan, inplace=True)


# In[8]:


data["solar_exposure"].mean()


# In[9]:


data["solar_exposure"].fillna(14.7,inplace=True)


# In[10]:


data["rainfall"].fillna(data["rainfall"].mode()[0], inplace=True)


# In[11]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['school_day'] = label_encoder.fit_transform(data['school_day'])
data['school_day'].unique()


# In[12]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['holiday'] = label_encoder.fit_transform(data['holiday'])
data['holiday'].unique()


# In[13]:


data['date'] = pd.to_datetime(data['date'])


# In[14]:


data['date']


# In[15]:


data.dtypes


# In[16]:


data.isnull().sum()


# In[17]:


data.columns


# In[18]:


#data.to_csv("Cleaned Electricity Demand Forecast")


# In[90]:


data1=pd.read_csv("Final Dataset")
data1


# # demand and RRP over time (daily, weekly, monthly).

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt

data1['date'] = pd.to_datetime(data1['date'])
data1.set_index('date', inplace=True)

data1['RRP'] = pd.to_numeric(data1['RRP'], errors='coerce')

daily = data1[['demand', 'RRP']].resample('D').mean()
weekly = data1[['demand', 'RRP']].resample('W').mean()
monthly = data1[['demand', 'RRP']].resample('M').mean()

plt.figure(figsize=(18, 20))


#Daily Plot
plt.subplot(3, 1, 1)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(daily.index, daily['demand'], label='Demand (Daily)', color='b')
ax2.plot(daily.index, daily['RRP'], label='RRP (Daily)', color='g')
ax1.set_title('Daily Demand and RRP')
ax1.set_xlabel('Date')
ax1.set_ylabel('Demand', color='b')
ax2.set_ylabel('RRP', color='g')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)

# Weekly plot
plt.subplot(3, 1, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(weekly.index, weekly['demand'], label='Demand (Weekly)', color='b')
ax2.plot(weekly.index, weekly['RRP'], label='RRP (Weekly)', color='g')
ax1.set_title('Weekly Demand and RRP')
ax1.set_xlabel('Date')
ax1.set_ylabel('Demand', color='b')
ax2.set_ylabel('RRP', color='g')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)

# Monthly plot
plt.subplot(3, 1, 3)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(monthly.index, monthly['demand'], label='Demand (Monthly)', color='b')
ax2.plot(monthly.index, monthly['RRP'], label='RRP (Monthly)', color='g')
ax1.set_title('Monthly Demand and RRP')
ax1.set_xlabel('Date')
ax1.set_ylabel('Demand', color='b')
ax2.set_ylabel('RRP', color='g')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)

plt.tight_layout()
plt.show()


# In[57]:


fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(data1.index, data1['demand'], label='Demand', color='b')
ax2 = ax1.twinx()
ax2.plot(data1.index, data1['RRP'], label='RRP', color='g')

ax1.set_xlabel('Date')
ax1.set_ylabel('Demand')
ax2.set_ylabel('RRP')

fig.legend(loc='upper left')
plt.title('Demand and RRP Over Time')
plt.grid(True)
plt.show()


# In[66]:


import pandas as pd
import matplotlib.pyplot as plt

data1['date'] = pd.to_datetime(data1['date'])

data1['RRP_positive'] = data1['RRP'].apply(lambda x: x if x > 0 else None)
data1['RRP_negative'] = data1['RRP'].apply(lambda x: x if x < 0 else None)

data1['RRP_positive_MA'] = data1['RRP_positive'].rolling(window=7, min_periods=1).mean()
data1['RRP_negative_MA'] = data1['RRP_negative'].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(14, 6))

# Positive Plot
plt.subplot(1, 2, 1)
plt.plot(data1['date'], data1['RRP_positive'], label='RRP Positive', color='green', alpha=0.6)
plt.plot(data1['date'], data1['RRP_positive_MA'], label='7-Day MA', color='darkgreen')
plt.title('Positive RRP Prices with Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Negative Plot
plt.subplot(1, 2, 2)
plt.plot(data1['date'], data1['RRP_negative'], label='RRP Negative', color='red', alpha=0.6)
plt.plot(data1['date'], data1['RRP_negative_MA'], label='7-Day MA', color='darkred')
plt.title('Negative RRP Prices with Moving Average')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[69]:


import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.regplot(x='RRP', y='demand', data=data1, scatter_kws={'s': 100, 'color': 'blue'}, line_kws={'color': 'red'})
plt.title('Demand vs. RRP Scatter Plot with Regression Line', fontsize=16)
plt.xlabel('Recommended Retail Price (RRP)', fontsize=14)
plt.ylabel('Demand', fontsize=14)
plt.show()


# # Weather Impact Analysis

# In[70]:


sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.regplot(x="min_temperature", y="demand", data=data1, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.title("Demand vs. Min Temperature")

plt.subplot(1, 2, 2)
sns.regplot(x="max_temperature", y="demand", data=data1, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.title("Demand vs. Max Temperature")

plt.tight_layout()
plt.show()


# In[72]:


# Line plots for seasonal patterns
plt.figure(figsize=(14, 8))
sns.lineplot(x="date", y="demand", data=data1, label="Demand")
sns.lineplot(x="date", y="solar_exposure", data=data1, label="Solar Exposure")
sns.lineplot(x="date", y="rainfall", data=data1, label="Rainfall")
plt.xticks(rotation=45)
plt.title("Demand, Solar Exposure, and Rainfall Over Time")
plt.legend()
plt.show()


# In[73]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = data1[["demand", "min_temperature", "max_temperature", "solar_exposure", "rainfall"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix for Demand and Weather Features")
plt.show()


# In[77]:


bins = pd.cut(data1["min_temperature"], bins=5)
data1["temp_bins"] = bins.astype(str)

g = sns.FacetGrid(data1, col="temp_bins", height=4, aspect=1.5)
g.map(sns.histplot, "demand")
g.set_titles("Demand Distribution for {col_name}")
g.set_axis_labels("Demand", "Frequency")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Demand Distribution Across Temperature Bins")
plt.show()


# # Operational Efficiency & Special Event Analysis

# In[78]:


average_demand = data1.groupby('holiday')['demand'].mean().reset_index()
average_demand['holiday'] = average_demand['holiday'].map({1: 'Holiday', 0: 'Regular Day'})

plt.figure(figsize=(10, 6))
sns.barplot(x='holiday', y='demand', data=average_demand, palette='viridis')

plt.title('Average Demand on Holidays vs. Regular Days')
plt.xlabel('Day Type')
plt.ylabel('Average Demand')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[79]:


df_melted = pd.melt(data1, id_vars=['demand'], value_vars=['holiday', 'school_day'], 
                    var_name='Day Type', value_name='Indicator')
df_melted['Indicator'] = df_melted['Indicator'].map({1: 'Yes', 0: 'No'})
plt.figure(figsize=(14, 8))
sns.boxplot(x='Day Type', y='demand', hue='Indicator', data=df_melted, palette='muted')
plt.title('Demand Distribution for Holidays vs. School Days')
plt.xlabel('Day Type')
plt.ylabel('Demand')
plt.legend(title='Is it a Holiday or School Day?')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Anomaly Detection and Outlier Analysis

# In[82]:



mean_demand = data1['demand'].mean()
std_demand = data1['demand'].std()

upper_control_limit = mean_demand + 3 * std_demand
lower_control_limit = mean_demand - 3 * std_demand

plt.figure(figsize=(14, 7))
plt.plot(data1['demand'], marker='o', label='Demand', color='b')
plt.axhline(mean_demand, color='g', linestyle='--', label='Mean Demand')
plt.axhline(upper_control_limit, color='r', linestyle='--', label='Upper Control Limit (UCL)')
plt.axhline(lower_control_limit, color='r', linestyle='--', label='Lower Control Limit (LCL)')

anomalies = data1[(data1['demand'] > upper_control_limit) | (data1['demand'] < lower_control_limit)]
plt.scatter(anomalies.index, anomalies['demand'], color='red', s=100, label='Anomalies')

plt.title('Control Chart for Demand with UCL and LCL')
plt.xlabel('Index')
plt.ylabel('Demand')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# # Revenue and Cost Optimization Opportunities

# In[83]:



plt.figure(figsize=(14, 8))
plt.stackplot(data1['date'], data1['demand_pos_RRP'], data1['demand_neg_RRP'],
              labels=['Positive RRP', 'Negative RRP'], colors=['#4caf50', '#f44336'], alpha=0.7)

plt.title('Stacked Area Plot: Positive vs. Negative RRP Over Time')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# # Advanced Temporal Feature Analysis

# In[84]:


#Autocorrelation Plot

demand_series = data1['demand']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sm.graphics.tsa.plot_acf(demand_series, lags=30, ax=axes[0], title="ACF Plot for Demand")
sm.graphics.tsa.plot_pacf(demand_series, lags=30, ax=axes[1], title="PACF Plot for Demand")
plt.show()


# In[88]:


#Seasonal Decomposition Plot
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

data1['date'] = pd.to_datetime(data1['date'])
data1.set_index('date', inplace=True)
data1 = data1.asfreq('D')  # Assuming daily frequency

stl = STL(data1['demand'], period=7, seasonal=13)  
result = stl.fit()

result.plot()
plt.suptitle('STL Decomposition of Demand')
plt.show()


# # Forecasting Readiness and Feature Engineering

# In[93]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


data1['year'] = pd.to_datetime(data1['date']).dt.year
data1['month'] = pd.to_datetime(data1['date']).dt.month
data1['day_of_week'] = pd.to_datetime(data1['date']).dt.dayofweek
data1['temp_range'] = data1['max_temperature'] - data1['min_temperature']
data1['avg_temperature'] = (data1['max_temperature'] + data1['min_temperature']) / 2

X = data1[['year', 'month', 'day_of_week', 'RRP', 'temp_range', 'avg_temperature']]
y = data1['demand']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
plt.title('Feature Importance After Engineering')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[94]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data1['year'] = pd.to_datetime(data1['date']).dt.year
data1['month'] = pd.to_datetime(data1['date']).dt.month
data1['day_of_week'] = pd.to_datetime(data1['date']).dt.dayofweek
data1['temp_range'] = data1['max_temperature'] - data1['min_temperature']
data1['avg_temperature'] = (data1['max_temperature'] + data1['min_temperature']) / 2

pairplot_features = ['demand', 'RRP', 'temp_range', 'avg_temperature', 'month']
sns.pairplot(data1[pairplot_features], diag_kind='kde', corner=True, plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Engineered Features', y=1.02)
plt.show()


# In[ ]:




