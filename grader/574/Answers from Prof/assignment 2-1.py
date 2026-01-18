#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[3]:


#Reading the data
airlineData = pd.read_csv('airline_costs_new.csv')


# In[4]:


#Checking it's head
airlineData.head()


# In[5]:


#Dropping null values
airlineData = airlineData.dropna()


# In[6]:


#scatter plot of length and population
plt.scatter(airlineData.Length, airlineData.Population)


# In[7]:


#Scatter plot of flight time and population
plt.scatter(airlineData.Daily_flight_time, airlineData.Population)


# In[8]:


#Creating the linear model and printing it's summary
lr_model = smf.ols(formula = 'Population ~ Length+Daily_flight_time', data = airlineData).fit()
lr_model.summary()


# In[9]:


#Creating the linear model and printing it's summary
lr_model2 = smf.ols(formula = 'Total_assets ~ Population', data = airlineData).fit()
lr_model2.summary()

