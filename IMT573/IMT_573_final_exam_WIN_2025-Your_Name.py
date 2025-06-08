#!/usr/bin/env python
# coding: utf-8

# # IMT 573 - Final Examination - YOUR NAME
# 
# Name: IMT 573 Teaching Team
# Last Updated: Mar 2025

# ### Instructions
# 
# This is a take-home final examination. You may use your computer, books/articles, notes, course materials, etc., but all work must be your own! Please review the course policy on AI-based learning tools. **References must be appropriately cited and you must make modifications on the code solutions found online, such as ChatGPT. Recycling online codes or other students' work is counted as plagiarism**. Please justify your answers and show all work; a complete argument must be presented to obtain full credit. Before beginning this exam, please ensure you have access to data programming environments used in the course; this can be on your own personal computer or on shared infrastructure hosted by the university. 
# 
# 1. Download the exam template notebook file from Canvas. Open exam notebook and supply your solutions to the exam by editing the notebook. 
# 
# 2. Be sure the exam contains your full name. 
# 
# 3. Be sure to include well-documented (e.g. commented) code chucks, figures, and clearly written text chunk explanations as necessary. Any figures should be clearly labeled and appropriately referenced within the text. Be sure that each visualization adds value to your written explanation; avoid redundancy -- you do not need four different visualizations of the same pattern.
# 
# 4.  **Collaboration is not allowed on this exam. You may only speak with the course instructor about this material.**
# 
# 5. All materials and resources that you use (with the exception of lecture slides) must be appropriately referenced within your assignment.
# 
# 6. Remember partial credit will be awarded for each question for which a serious attempt at finding an answer has been shown. Students are **strongly** encouraged to attempt each question and to document their reasoning process even if they cannot find the correct answer. If you would like to include code to show this process, but it does not run without errors, you can do so by commenting out that code. 
# 
# 7. When you have completed the assignment and have **checked** that your code both runs in the Console and compiles correctly rename the file to `YourLastName_YourFirstName.ipynb`, and submit BOTH your HTML and notebook files on Canvas.

# ### Statement of Compliance
# 
# You **must** include the a "signed" Statement of Compliance in your submission. The Compliance Statement is found below. You must include this text, word-for-word, in your final exam submission. Adding your name indicates you have read the statement and agree to its terms. Failure to do so will result in your exam **not** being accepted.
# 
# **Statement of Compliance**
# 
# I affirm that I have had no conversation regarding this exam with any persons other than the instructor. Further, I certify that the attached work represents my own thinking. Any information, concepts, or words that originate from other sources are cited in accordance with University of Washington guidelines as published in the Academic Code (available on the course website). I am aware of the serious consequences that result from improper discussions with others or from the improper citation of work that is not my own. 
# 
# (you name goes here as signature) 
# 
# (date of signature)

# In[ ]:





# For this assignment, you'll need (at least) the following packages. If the package does not load, be sure it is properly installed.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# ### Problem 1
# 
# Points: 20
# 
# In this problem we will use data on infidelity, known as the Fair's Affairs dataset. The `Affairs` dataset is available as part of the `datasets` package in`statsmodels`. The original data come from two survey conducted by Psychology Today in 1969 and Redbook in 1974, see Greene (2003) and Fair (1978) for more information. We will use the Redbook data here.
# 
# The dataset contains various self-reported characteristics of 6,366 participants, including how often the respondent engaged in extramarital sexual intercourse during the past year, as well as their age, year married, whether they had children, their religiousness (on a 5-point scale, from 1=anti to 5=very), education, occupation (Hillingshead 7-point classification with reverse numbering), and a numeric self-rating of their marriage (from 1=very unhappy to 5=very happy).

# <font color = red>SOLUTION: 

# In[ ]:


data = sm.datasets.fair.load_pandas()
print(sm.datasets.fair.SOURCE)
print(sm.datasets.fair.NOTE)

affairs = data.data


affairs.head
# affairs.describe()


# (a) Describe the participants. Use descriptive, summarization, and exploratory techniques (visualizations) to describe the participants in the study. Please answer questions :  What is the average age of respondents? What is the proportion of families who do not have any children ? 

# In[ ]:





# (b) Suppose we want to explore the characteristics of participants who engage in extramarital sexual intercourse (i.e. affairs). Instead of modeling the number of affairs, consider the binary outcome - had an affair versus didn't have an affair. Create a new variable to capture this response variable of interest. What might the advantages and disadvantages of this approach to modeling the data be in this context?

# In[ ]:





# (c) Use an appropriate regression model to explore the relationship between having an affair (binary) and other personal characteristics. 

# In[ ]:





# (d) Interpret question (c)'s model results.

# In[ ]:





# (e) Create an artificial test dataset where martial rating varies from 1 to 5 and all other variables are set to their means. Use it as a test dataset to obtain predicted "affair" based on question (c)'s best model for cases in the test data. Interpret your results and use a visualization to support your interpretation.
# 
# Remember: if you notice any variables which do not make statistically significant impacts on the response variable based on the part (c) model summary, then remove them to retrieve the best model

# In[ ]:





# (f) Use a stepwise selection procedure (forward selection or backward elimination) to obtain a "best" regression model between this response variable - affairs(measure of time spent in extramarital affairs) and other characteristics. Is the "best" model different from the best model you got from question (c)? Which variables are included in this question's "best" fit model?

# In[ ]:





# (g) Reflect on your analysis in this problem. After completing all the parts of this analysis what remaining and additional ethical and privacy conerns do you have?

# In[ ]:





# ### Problem 2
# 
# Points: 20
# 
# In this problem set, we will use some data from a sports context. The data is provided as part of the [Introduction to Statistial Learning with Applications in Python](https://www.statlearning.com/) textbook. It was taken from the StatLib library which is maintained at Carnegie Mellon University. The data provided is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.
# 
# Following the directions below to load this data, `Hitters` directly from the `ISLP` python package. You may need to install the `ISLP` package before you can get started. In the Jupyter Hub you can open a terminal and use `pip install ISLP`. Next load the data as follows:

# In[ ]:


from ISLP import load_data
Hitters = load_data('Hitters')
Hitters.head()


# (a) Develop your own question to address in this analysis. Your question should be specific and measurable, and it should be able to be addressed through a basic analysis of the `Hitters` dataset. Hint: you will need to get to know this dataset and the variables available to formulate an appropriate question.

# In[ ]:





# (b) Briefly summarize the dataset, describing what data exists and its basic properties. Comment on any issues that need to be resolved before you can proceed with your analysis. 
# 

# In[ ]:





# (c) Use the dataset to provide empirical evidence that addressed your question from (a). Discuss your results. Provide **at least two visualizations** to support your story. 

# In[ ]:





# (d) Comment the questions (and answers) in this analysis.  Were you able to answer all of these questions?  Are all questions well defined?  Is the data good enough to answer all these?

# In[ ]:





# ### Problem 3
# 
# Points: 20
# 
# In this problem, we will use the 'sales' dataset. 
# 
# The dataset describes over 900 territories' car sales and cost:
# - territory_id: ID of the territory
# - product: car brand
# - sales: the unique territory's car sales voume
# - cost: the unique territory's car costs, inccluding shipping, marketing and customer support. 
# 
# In our analysis, we will be interested in the relationship between `sales` and `costs`.
# 
# (a) Use exploratory techniques to summarize the variables in the datasets (at least 2 visualizations)
# 

# In[ ]:


sales = pd.read_csv('sales.csv')
sales.head()


# In[ ]:





# (b) Calculate the Pearson correlation score bewteen 'sales' and 'cost' based on the score formula step by step. 

# ##### Please do NOT use any built-in packages or libraries to populate the correlation score; instead, use the formulas stated below to solve the problem
# 
# The Pearson correlation coefficient, $r$, between two variables $X1$ and $X2$ is defined as:
# 
# $$ r = \frac{\sum_{i=1}^n (X1_i - \overline{X1})(X2_i - \overline{X2})}{\sqrt{\sum_{i=1}^n (X1_i - \overline{X1})^2} \sqrt{\sum_{i=1}^n (X2_i - \overline{X2})^2}} $$
# 
# Where:
# - $X1_i$ and $X2_i$ are the values of the $i^{th}$ data point of variables $X1$ and $X2$, respectively.
# - $\overline{X}$ and $\overline{X2}$ are the means of $X1$ and $X2$.
# 

# In[ ]:





# In[ ]:





# (c) Train a simple linear regression model on response variable -'sales' and predictor variable -'costs'.  Calculate the coefficients (slope and intercept).

# ##### Please do NOT use any packages or libraries to get the coefficient values; instead, use the formulas stated below to proceed.

# In a simple linear regression model where $y$ (sales) is the response variable and $x$ (cost) is the predictor variable, the slope ($\beta_1$) and intercept ($\beta_0$) are given by:
# 
# ##### Slope ($\beta_1$):
# $$ \beta_1 = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sum_{i=1}^n (x_i - \overline{x})^2} $$
# 
# ##### Intercept ($\beta_0$):
# $$ \beta_0 = \overline{y} - \beta_1 \overline{x} $$
# 
# Where:
# - $x_i$ and $y_i$ are the values of the $i^{th}$ data point for the variables $x$ and $y$, respectively.
# - $\overline{x}$ and $\overline{y}$ are the means of $x$ and $y$.
# 

# In[ ]:





# In[ ]:





# (d) Write down the simple linear regression model equation based on (c) results. Provide reflections on the model equation.

# In[ ]:





# In[ ]:





# ### Problem 4
# 
# Points: 20
# 
# The Wisconsin Breast Cancer dataset is available as a comma-delimited text file on the UCI Machine Learning Repository {https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original}. Our goal in this problem will be to predict whether observations (i.e. tumors) are malignant or benign. We will use the original dataset in this problem. 

# (a) Obtain the data, and load it into your programming environment by pulling it directly from the web. (Do **not** download it and import it from any CSV file.) Give a brief description of the data. 

# The Wisconsin Breast Cancer dataset contains 699 samples and a total of 11 variables. Some data samples have missing data. 

# In[ ]:


# Web location of data file
loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds = "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
url = loc + ds

# Load data
breast_data = pd.read_csv(url, sep=",", header=None, na_values="?")
breast_data.head


# In[ ]:





# In[ ]:





# (b) Tidy the data, ensuring that each variable is properly named and cast as the correct data type. Discuss any missing data.

# In[ ]:





# (c) Split the data into a training and validation set such that a random 70\% of the observations are in the training set.

# In[ ]:





# (d) Fit a machine learning model to predict whether tissue samples are malignant or benign. Compute and discuss the resulting confusion matrix. Be sure to address which of the errors that are identified you consider most problematic in this context.

# In[ ]:





# ### Problem 5
# 
# Points: 10
# 
# Please answer the questions below by writing a short response. 

# (a) Please describe 3 **classification** machine learning methods and each model's real world application.

# In[ ]:





# b) This quarter, we discussed four 'Missing data' types: 1) Not missing at random (NMAR), 2) Missing completely at random(MCAR), 3) Missing at random(MAR)and 4) Missing by design (MD).  Select **2 missing data types** listed above. Describe those two missing data types meanings and provide 1 real world example for each missing data type.

# In[ ]:





# (c) What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?

# In[ ]:





# ### Problem 6
# 
# Points: 10
# 
# Suppose we have a dataset with five predictors, $X_1 =$ GPA, $X_2 =$ IQ, $X_3 =$ Race (1 for Female, and 0 for Male), $X_4 =$ Interaction between GPA and IQ, and $X_5 =$ Interaction between GPA and Gender. Note: the data here is limited as gender was collected as a binary variable.
# 
# The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model and get $\hat{\beta}_0=50, \hat{\beta}_1=20, \hat{\beta}_2=0.07, \hat{\beta}_3=35, \hat{\beta}_4=0.01$, and $\hat{\beta}_5=-10$. 

# (a) Which answer is correct and why? <font color = red>[single choice question]
# 
# i. For a fixed value of IQ and GPA, males earn more on average than females.
# 
# ii. For a fixed value of IQ and GPA, females earn more on average than males.
# 
# iii. For a fixed value of IQ and GPA, males earn more on average than females provided that the GPA is high enough.
# 
# iv. For a fixed value of IQ and GPA, females earn more on average than males provided that the GPA is high enough.

# In[ ]:





# (b) Predict the salary of a female with IQ of 110 and a GPA of 4.0.

# In[ ]:





# (c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is little evidence of an interaction effect. Justify your answer.

# In[ ]:





# ### Problem 7 - Extra Credit
# 
# Points: Max. 5
# 
# Suppose that $X_1, \ldots X_n$ form a random sample from a Poisson distribution for which the mean $\theta$ is unknown, ($\theta>0)$.

# (a) Determine the MLE of $\theta$, assuming that at least one of the observed values is different from 0. Show your work.

# In[ ]:





# (b) Show that the MLE of $\theta$ does not exists if every observed value is 0.

# In[ ]:





# ### Problem 8 - Extra Credit 
# 
# Points: Max. 5 
# 
# 7 Democratic party members and 7 Republican party members are running for 5 seats (1 president, 1 vice president and 3 advisors) in a committee. Voters choose 5 people at random uniformly. 
# 
# What is the probability of this event, in which Frank (Democratic) becomes the president and Will( Republican) becomes the vice-president ?

# In[1]:


from IPython.display import Image
Image("Extra.png")


# (a) Write down your solution logics. (We want to see your reasonings : what makes the sample space (or denominator )and what makes the numerator? Why you think so ?

# In[ ]:





# (b) Write down your calculations for the probability. [Hint: Think about whether the 2 roles (president and VP) assignment is ordered or not. Do not trust LLMs such as ChatGPT's answer completely ]

# In[ ]:




