#!/usr/bin/env python
# coding: utf-8

# # IMT 573 - Final Examination - Steve Gonzales
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
# Steve Gonzales  Mar 12, 2025

# For this assignment, you'll need (at least) the following packages. If the package does not load, be sure it is properly installed.

# In[4]:


import os, re, math, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# ### Problem 1
# 
# Points: 20
# 
# In this problem we will use data on infidelity, known as the Fair's Affairs dataset. The `Affairs` dataset is available as part of the `datasets` package in`statsmodels`. The original data come from two survey conducted by Psychology Today in 1969 and Redbook in 1974, see Greene (2003) and Fair (1978) for more information. We will use the Redbook data here.
# 
# The dataset contains various self-reported characteristics of 6,366 participants, including how often the respondent engaged in extramarital sexual intercourse during the past year, as well as their age, year married, whether they had children, their religiousness (on a 5-point scale, from 1=anti to 5=very), education, occupation (Hillingshead 7-point classification with reverse numbering), and a numeric self-rating of their marriage (from 1=very unhappy to 5=very happy).

# <font color = red>SOLUTION: 

# In[5]:


data = sm.datasets.fair.load_pandas()
print(sm.datasets.fair.SOURCE)
print(sm.datasets.fair.NOTE)
affairs = data.data
affairs.head
# affairs.describe()


# (a) Describe the participants. Use descriptive, summarization, and exploratory techniques (visualizations) to describe the participants in the study. Please answer questions :  What is the average age of respondents? What is the proportion of families who do not have any children ? 

# In[6]:


#Basic descriptive analysis of data
columns = affairs.columns
unique_values = {col: affairs[col].unique() for col in columns}
for col, values in unique_values.items():
    print(f"Unique values for {col}: {values[:9]}")


# In[7]:


# Answer to Question a
# Descriptive information about the Dataset
print("Answer to Question 1a Descriptive information about the Dataset")
display(affairs.isna().sum())
print("Shape: ", affairs.shape)
affairs.describe()


# There are several discrete variables, so we will convert them to dummies, "one hot" when we fit the models.

# <b>Answer:</b><p>
# The data seems to look good: all categories are adhered to and there are no NaN values

# In[8]:


# Answer to Question a
# Summary information about the Dataset
df_affairs = affairs.copy()
print("Answer to Question 1a Summary information about the Dataset")
n_total_respondents = len(df_affairs)
print(f"There are a total of {n_total_respondents:,} respondents")
print(f"The average age of respondents: {df_affairs['age'].mean():.2f}")
n_have_no_child = len(affairs[(df_affairs['children'] == 0)])
n_have_child = len(affairs[(df_affairs['children'] != 0)])
print(f"Proportion of families who do not have any children: {n_have_no_child/n_total_respondents:.2%}")


# In[9]:


# Answer to Question a
# Summary information about the Dataset
print("Answer to Question 1a Exploratory information about the Dataset")
# Basic Distribution of Variables

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, col in enumerate(columns):
    if col != 'affairs':
        # Plot histograms for each column
        sns.histplot(affairs[col], ax=axes[i], binwidth=1)
        axes[i].set_title(f'Dist. of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

# Remove any empty subplots
for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[10]:


# add a visualization


# (b) Suppose we want to explore the characteristics of participants who engage in extramarital sexual intercourse (i.e. affairs). Instead of modeling the number of affairs, consider the binary outcome - had an affair versus didn't have an affair. Create a new variable to capture this response variable of interest. What might the advantages and disadvantages of this approach to modeling the data be in this context?

# In[11]:


# Create a new variable where the length of the affair is greater than 0
df_affairs['had_affair'] = (df_affairs['affairs'] > 0).astype(int)
display(df_affairs)


# In[12]:


# Since we encountered divide by zero, let's take log + 1
df_affairs['log_affairs'] = df_affairs['affairs'].apply(np.log1p)
affairs['log_affairs'] = affairs['affairs'].apply(np.log1p)
display(df_affairs)


# <b>Answer:</b><p>
# The advantage to creating `df['had_affair']` is that we can more accurately predict infidelity with this variable. Allows the use of classification algorithms which are well-suited for binary outcomes.<p>
# However, the disadvantage is that if we wanted to predict the extent of infidelity, we obviously could not use this variable. It also removes some data that might be useful for other correlation analysis. In addition, there are many kinds of models that should not be used to predict a binary outcome.

# (c) Use an appropriate regression model to explore the relationship between having an affair (binary) and other personal characteristics. 

# In[20]:


# Will use Logistic Regression model
# Use statsmodels built in C() to handle the categorical values
categorical_columns = ['educ', 'occupation', 'occupation_husb']

# Create the formula string for statsmodels
formula = 'had_affair ~ ' + ' + '.join([f'C({col})' if col in categorical_columns else col for col in df_affairs.columns if col not in ['log_affairs', 'affairs', 'had_affair']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_affairs, df_affairs['had_affair'], test_size=0.3, random_state=42)

# Fit the logistic regression model using statsmodels
logit_model = smf.logit(formula, data=X_train).fit()

# Make predictions on the test set
y_pred_probs = logit_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Affair', 'Affair'])
disp.plot(cmap='Blues', values_format='d', colorbar=False)
plt.title('Logit Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['No Affair', 'Affair'])
print(report)

# Extract accuracy from classification report
report_str = classification_report(y_test, y_pred, output_dict=False)
match = re.search(r'accuracy\s+(\d+\.\d+)', report_str)
accuracy = -1
if match:
    accuracy = float(match.group(1))

print(f"Accuracy: {accuracy}")


# In[14]:


# Function to return the significant and non-significant P-Values

def evaluate_pvalues(model_summary, threshold=0.05):
    """
    Evaluate the P-Values from a statsmodel model summary and output the column names
    that have P-Values below/above the given threshold.
    Parameters:
    model_summary (sm.iolib.summary.Summary): The summary object of the fitted model.
    threshold (float): The threshold for P-Values. Default is 0.05.
    Returns:
    dict: A dictionary with column names as keys and their P-Values as values.
    """
    p_values = model_summary.tables[1].data[1:]  # Extract the P-Values table
    sig, non_sig = {}, {}
    for row in p_values:
        column_name = row[0]
        if column_name.lower() not in ['const', 'intercept']:
            pvalue = float(row[4])
            if pvalue > threshold:
                non_sig[column_name] = pvalue
            else:
                sig[column_name] = pvalue
    return sig, non_sig

# Example usage:
# model_summary = result.summary()
# sig, non_sig = evaluate_pvalues(model_summary, threshold=0.05)
# print(f"P Values above threshold (ie non significant): {non_sig}")


# In[62]:


# Logistic Regression: Calculate P-Values to find a "best model" (will need for c)
print("--- Logit Model with C() ---")
logit_formula = """had_affair ~ rate_marriage + age + yrs_married + children + religious + 
C(educ) + C(occupation) + C(occupation_husb)"""
logit_model_with_c = smf.logit(logit_formula, data=df_affairs).fit()
logit_model_with_c_summary = logit_model_with_c.summary()
print(logit_model_with_c_summary)
keep_variables, drop_variables = evaluate_pvalues(logit_model_with_c_summary, threshold=0.05)
print(f"Variables to drop due to not being statistically significant: {drop_variables}")
print(f"Variables to keep; statistically significant: {keep_variables}")


# The "Best Model" for this should exlude `educ` `children` and `occupation_husb` because they are not statistically significant. Some levels of Occupation however, can be included. So we keep these:
# `'C(occupation)[T.5.0]': 0.014, 'C(occupation)[T.6.0]': 0.013, 'rate_marriage': 0.0, 'age': 0.0, 'yrs_married': 0.0, 'religious': 0.0`

# In[63]:


# Function to evaluate p-values and determine which variables to keep
def evaluate_pvalues(summary, threshold=0.05):
    pvalues = summary.tables[1].data[1:]
    keep_variables = []
    drop_variables = []
    for row in pvalues:
        variable = row[0]
        pvalue = float(row[4])
        if pvalue <= threshold:
            keep_variables.append(variable)
        else:
            drop_variables.append(variable)
    return keep_variables, drop_variables

# Get variables to keep and drop
keep_variables, drop_variables = evaluate_pvalues(logit_model_with_c_summary, threshold=0.05)
print(f"Variables to drop due to not being statistically significant: {drop_variables}")
print(f"Variables to keep; statistically significant: {keep_variables}")

# Construct new formula with significant variables
new_formula = "had_affair ~ " + " + ".join(keep_variables)
print(f"New formula: {new_formula}")

# Fit the new logistic regression model
new_logit_model = smf.logit(new_formula, data=df_affairs).fit()
new_summary = new_logit_model.summary()
print(new_summary)


# In[60]:


from patsy.contrasts import Treatment  
# Logistic Regression: Calculate P-Values to find a "best model" (will need for c)
print("--- BEST Logit Model with C() ---")
best_logit_formula = """had_affair ~ rate_marriage + yrs_married + religious + \
C(occupation, Treatment(reference='managerial_administrative_business')) + \
C(occupation, Treatment(reference='professional')) + \
C(occupation, Treatment(reference='white_collar'))"""
best_logit_model_with_c = smf.logit(best_logit_formula, data=df_affairs).fit()
summary = best_logit_model_with_c.summary()
print(summary)
keep_variables, drop_variables = evaluate_pvalues(summary, threshold=0.05)
print(f"Variables to drop due to not being statistically significant: {drop_variables}")
print(f"Variables to keep; statistically significant: {keep_variables}")


# In[46]:


def get_logit_model(X, y):
    # Add a constant to the predictor variables (intercept)
    X = sm.add_constant(X)

    # Fit the logistic regression model
    logit_model = sm.Logit(y, X)
    logit_model_fit = logit_model.fit()

    return logit_model_fit.summary(), logit_model_fit


# (d) Interpret question (c)'s model results.

# <b>Answer:</b><p>
# By fitting the model and looking at the P-Values, we exlude `children` and `occupation_husb` and some occupations because they are not statistically significant.<p>
# We will move forward with <b>Logistic Regression Best Model</b> with `'C(occupation)[T.5.0]': 0.014, 'C(occupation)[T.6.0]': 0.013, 'rate_marriage': 0.0, 'age': 0.0, 'yrs_married': 0.0, 'religious': 0.0`<p>
# We see that `rate_marriage` has a significant negative correlation to whether or not an affair took place, which is expected.<p>
# The strongest positive correlation is between Occupation, 5 = managerial, administrative, business, 6 = professional with advanced degree.

# (e) Create an artificial test dataset where marital rating varies from 1 to 5 and all other variables are set to their means. Use it as a test dataset to obtain predicted "affair" based on question (c)'s best model for cases in the test data. Interpret your results and use a visualization to support your interpretation.
# 
# Remember: if you notice any variables which do not make statistically significant impacts on the response variable based on the part (c) model summary, then remove them to retrieve the best model

# First create dummies (one hot) for categorical variables so its easier to calculate the mean.

# In[17]:


# Function to create dummies (one hot) from columns
def create_dummies(df, categorical_variables):
    dummies = []
    for col in categorical_variables:
        dummies.append(pd.get_dummies(df[col], prefix=col))
    
    df = pd.concat([df] + dummies, axis=1)
    # Convert the Booleans to 1 or 0
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    # Drop the original columns
    df = df.drop(categorical_variables, axis=1)
    return df
# Function to shorten the col names
def process_string(s):
    # Replace anything not a lowercase letter with a space
    s = re.sub(r'[^a-z]', ' ', s)
    # Deduplicate spaces
    s = re.sub(r'\s+', ' ', s)
    # Replace all single spaces with an underscore
    s = s.replace(' ', '_')
    return s


# In[47]:


# Map the categorical data to dummy columns
maps = {
    'educ': {
        9: 'grade school',
        12: 'high school',
        14: 'some college',
        16: 'college graduate',
        17: 'some graduate school',
        20: 'advanced degree'
    },
    'occupation': {
        1: 'student',
        2: 'farming, ag',
        3: 'white-collar',
        4: 'teacher, counselor',
        5: 'managerial, administrative, business', # Keep
        6: 'professional' # Keep
    },
    'occupation_husb': {
        1: 'student',
        2: 'farming, ag',
        3: 'white-collar',
        4: 'teacher, counselor',
        5: 'managerial, administrative, business',
        6: 'professional'
    }
 }
df_affairs = affairs.copy()
df_affairs['had_affair'] = (df_affairs['affairs'] > 0).astype(int)
df_affairs['log_affairs'] = df_affairs['affairs'].apply(np.log1p)
for col_name in maps.keys():
    new_map = {k: process_string(v) for k, v in maps[col_name].items()}
    df_affairs[col_name] = df_affairs[col_name].map(new_map)

display(df_affairs.head())
df_affairs_with_dummies = create_dummies(df_affairs, ['educ', 'occupation', 'occupation_husb'])
display(df_affairs_with_dummies.head())


# Use only the predictors with a lower than .05 P-Value:<p>
#  - 'C(occupation)[T.managerial_administrative_business]': 0.0,
#  - 'C(occupation)[T.professional]': 0.002, 
#  - 'C(occupation)[T.white_collar]': 0.003, 
#  - 'rate_marriage': 0.0, 
#  - 'age': 0.0, 
#  - 'yrs_married': 0.0, 
#  - 'religious': 0.0}

# In[48]:


# Use only the predictors with a lower than .05 P-Value
# {'C(occupation)[T.managerial_administrative_business]': 0.0, 'C(occupation)[T.professional]': 0.002, 'C(occupation)[T.white_collar]': 0.003, 'rate_marriage': 0.0, 'age': 0.0, 'yrs_married': 0.0, 'religious': 0.0}
best_ppredictors = ['rate_marriage', 'age', 'yrs_married', 'religious',
                     'occupation_managerial_administrative_business', 'occupation_professional', 
                     'occupation_white_collar']
result_e, best_fit_model_e = get_logit_model(df_affairs_with_dummies[best_ppredictors], df_affairs_with_dummies['had_affair'])
print(result_e)


# In[49]:


# Create a new DF with average values for all vars except the affairs columns
variables_to_mean = ['age', 'yrs_married', 'religious',
                     'occupation_managerial_administrative_business', 'occupation_professional', 
                     'occupation_white_collar']
# Create a new DF
df_affairs_means = df_affairs_with_dummies[variables_to_mean].copy()

# Create a single DF with one row per column of means
df_means = df_affairs_means.mean()
print("Average (Mean) Values:")
display(df_means)

for col in df_affairs_means.columns:
    if col not in ['rate_marriage', 'affairs', 'log_affairs', 'had_affair']:
        df_affairs_means.loc[:, col] = df_means[col]
# Add back in the non-simulated data (ie real)
additional_cols = ['had_affair', 'rate_marriage']
for col in additional_cols:
    df_affairs_means.loc[:, col] = df_affairs[col].values
display(df_affairs_means)


# In[51]:


# Predict using the Best Model
simulated_predictions = best_fit_model_e.predict(df_affairs_means)
df_affairs_means_predicted = df_affairs_means.copy()
df_affairs_means_predicted['predicted_affair'] = simulated_predictions.apply(lambda x: '{:.4f}'.format(x))
display(df_affairs_means_predicted)


# In[52]:


simulated_predictions.describe()


# Looking at the data with a cutoff, to see if there are any valid predictions.

# In[ ]:


# Predict using the Best Model
predictions_binary = (best_logit_result.predict(df_affairs_means) > 0.5).astype(int)
df_affairs_means_predicted = df_affairs_means.copy()
df_affairs_means_predicted['predicted_affair'] = predictions_binary
display(df_affairs_means_predicted[(df_affairs_means_predicted['predicted_affair'] > 0)])


# Once we converted all but one of the variables to the mean, we have taken away most of the prediction power of the model.
# The model (with the means) will tend to predict probabilities closer to the overall average probability of the outcome, because the averaged variables provide no unique information.
# 
# 

# (f) Use a stepwise selection procedure (forward selection or backward elimination) to obtain a "best" regression model between this response variable - affairs (measure of time spent in extramarital affairs) and other characteristics. Is the "best" model different from the best model you got from question (c)? Which variables are included in this question's "best" fit model?

# <b>Answer:</b><p>
# <b>Going back to the real data set without the means (as per Prof Sophin)</b><p>
# When using stepwise selection, we can reject any variable whose p-value is above .05. Also, when adding a variable in, if it increases the Adjusted R-squared value, we know it is improving the regression fit.

# In[ ]:


# Going back to the regular data set
# This will figure out what combination of predictors are best
predictors = ['rate_marriage', 'age', 'yrs_married', 'religious', 'educ', 'occupation']
y = affairs['affairs']

# Set a minimum number for this tracking variable
prev_adj_r = -999.99

for i in range(1, len(predictors) + 1):
    print(f"Trying {predictors[:i]} with OLS")
    X = sm.add_constant(affairs[predictors[:i]])
    model = sm.OLS(y, X)
    result = model.fit(maxiter=1000)
    # print(result.summary())
    p_values = result.pvalues.round(8)
    adj_r_squared = result.rsquared_adj
    print(f'Adjusted R-Squared: {adj_r_squared:.6f}')
    if adj_r_squared > prev_adj_r:
        print("***better adjusted r squared")
    prev_adj_r = adj_r_squared
    good_p_values = p_values[p_values <= .05]
    print(good_p_values.to_string())
    print('='*50)
    print('')


# <b>Answer:</b><p>
# We can't use the Logistic Regression model from Answer 1e because now we are predicting a continuous values. In this case we use Ordinary Least Squares.<p>
# As you can see from above, with each addition of a new variable, the adjusted R Squared increases until just before `educ` is added in, but then with all the variables, the Adjusted R Square is the highest.<p>

# In[ ]:


# Show will all variables
predictors = ['rate_marriage', 'age', 'yrs_married', 'religious', 'educ', 'occupation']
y = affairs['affairs']

print(f"Trying {predictors} with OLS")
X = sm.add_constant(affairs[predictors])
model = sm.OLS(y, X)
result = model.fit(maxiter=1000)
print(result.summary())


# We will want to use the last one with the variables: `['rate_marriage', 'yrs_married', 'religious', 'occupation']` we will remove `age` and `educ` because they are not statistically significant.<p>

# In[ ]:


# Trying with the subset of predictors
predictors = ['rate_marriage', 'yrs_married', 'religious', 'occupation']
y = affairs['affairs']

print(f"Best Fit {predictors} with OLS")
X = sm.add_constant(affairs[predictors])
model = sm.OLS(y, X)
result = model.fit(maxiter=1000)
print(result.summary())


# `occupation` is not statistically significant.<p>
#  <b>The following is the best fit model:</b>

# In[ ]:


predictors = ['rate_marriage', 'yrs_married', 'religious']
y = affairs['affairs']

print(f"Best Fit {predictors} with OLS")
X = sm.add_constant(affairs[predictors])
model = sm.OLS(y, X)
result = model.fit(maxiter=1000)
print(result.summary())


# (g) Reflect on your analysis in this problem. After completing all the parts of this analysis what remaining and additional ethical and privacy conerns do you have?

# <b>Answer:</b><p>
# TBD: Eithical and Privacy

# ### Problem 2
# 
# Points: 20
# 
# In this problem set, we will use some data from a sports context. The data is provided as part of the [Introduction to Statistial Learning with Applications in Python](https://www.statlearning.com/) textbook. It was taken from the StatLib library which is maintained at Carnegie Mellon University. The data provided is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.
# 
# Following the directions below to load this data, `Hitters` directly from the `ISLP` python package. You may need to install the `ISLP` package before you can get started. In the Jupyter Hub you can open a terminal and use `pip install ISLP`. Next load the data as follows:

# In[ ]:


try:
    from ISLP import load_data
    Hitters = load_data('Hitters')
except:
    print("This package is old and cannot be installed on Windows, data copied from Github instead")
    Hitters = pd.read_csv('Hitters.csv')
    
Hitters.head()
print(Hitters.shape)


# <b>Note:</b><p>
# Cannot install ISLP; I tried for hours. There is an issue with versions and Cargo and Rust, which are massive platforms. The CSV files are quite small and readily available on Github.
# ![image.png](attachment:image.png)

# <b>Answer:</b><p>
# Major League Baseball Data from the 1986 and 1987 seasons.
# - `AtBat`: Number of times at bat in 1986
# - `Hits`: Number of hits in 1986
# - `HmRun`: Number of home runs in 1986
# - `Runs`: Number of runs in 1986
# - `RBI`: Number of runs batted in in 1986
# - `Walks`: Number of walks in 1986
# - `Years`: Number of years in the major leagues
# - `CAtBat`: Number of times at bat during his career
# - `CHits`: Number of hits during his career
# - `CHmRun`: Number of home runs during his career
# - `CRuns`: Number of runs during his career
# - `CRBI`: Number of runs batted in during his career
# - `CWalks`: Number of walks during his career
# - `League`: A factor with levels A and N indicating player’s league at the end of 1986
# - `Division`: A factor with levels E and W indicating player’s division at the end of 1986
# - `PutOuts`: Number of put outs in 1986
# - `Assists`: Number of assists in 1986
# - `Errors`: Number of errors in 1986
# - `Salary`: 1987 annual salary on opening day in thousands of dollars
# - `NewLeague`: A factor with levels A and N indicating player’s league at the beginning of 1987<p>
# The salary data were originally from Sports Illustrated, April 20, 1987.

# (a) Develop your own question to address in this analysis. Your question should be specific and measurable, and it should be able to be addressed through a basic analysis of the `Hitters` dataset. Hint: you will need to get to know this dataset and the variables available to formulate an appropriate question.

# In[ ]:


# Basic descriptive analysis
display(Hitters.isna().sum())
print(Hitters.shape)
Hitters.describe()


# <b>Question:</b> Are Home Runs a predictor of overall Runs? Ie Does the player who hits more home runs, also make it to home plate when other players are batting? Or possibly, are they a poor runner and the two variables would be negatively correlated. Will do this for 1986 only (not career). 

# (b) Briefly summarize the dataset, describing what data exists and its basic properties. Comment on any issues that need to be resolved before you can proceed with your analysis. 
# 

# The data consists of mostly data from the 1986 season, with some information on 1987 and most 1986 statistics are accompanied by career statistics. The data is mostly clean except for Salary information which has about 20% of the data missing: my original question was, "Does Salary predict Runs or Home Runs?", but I don't think it's appropriate with this dataset.<p>
# For ease of processing, I will create a new column that subtracts out Home Runs from Runs, which will equal the player's runs attributed to non-Home Runs.

# (c) Use the dataset to provide empirical evidence that addressed your question from (a). Discuss your results. Provide **at least two visualizations** to support your story. 

# In[ ]:


# Prepare the dataframe by creating a new column which has Runs not attributed to Home Runs
Hitters['nonHRRuns'] = Hitters['Runs'] - Hitters['HmRun']
# Prepare the dataframe by converting NaN to 0
Hitters.fillna(0, inplace=True)
display(Hitters)


# <b>Answer:</b><p>
# Hypothesis is made that Non Home Run Runs (nonHRRuns) are more predictive than Total Runs (Runs). Let's test that hypothesis.

# In[ ]:


fig, axes = plt.subplots(2, 1, figsize=(15, 14))
axes = axes.flatten()
# Filter out Runs, because it is just a proxy for nonHRRuns, filter out text data
predictors = ['AtBat', 'Hits']
y_vals = ['nonHRRuns', 'Runs']
for i, col in enumerate(predictors):
    for y_val in y_vals:
        # Create a scatter plot with a regression line
        sns.regplot(x=col, y=y_val, data=Hitters, ax=axes[i], scatter_kws={'s': 10}, label=y_val) #scatter_kws to make points smaller.
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(f"Qty of Runs (both HR and regular)")
        axes[i].set_title(f"{y_val} vs. {col}")
        axes[i].grid(True)
    axes[i].legend()
plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 14))
axes = axes.flatten()
# Filter out Runs, because it is just a proxy for nonHRRuns, filter out text data
predictors = ['AtBat', 'Hits']
y_vals = ['nonHRRuns', 'Runs']
i = 0
for col in predictors:
    for y_val in y_vals:
        # Create a scatter plot with a regression line
        if y_val == 'nonHRRuns':
            color = 'blue'
        else:
            color = 'orange'
        sns.regplot(x=col, y=y_val, data=Hitters, ax=axes[i], scatter_kws={'s': 10}, color=color) #scatter_kws to make points smaller.
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(f"Qty of {y_val}")
        axes[i].set_title(f"{y_val} vs. {col}")
        axes[i].grid(True)
        i += 1
plt.tight_layout()
plt.show()


# <b>Answer:</b><p>
# Runs have a few more extreme outliers, but both seem to correlate similarly to the prediction variables.

# In[ ]:


# Visualize a regression fit for all the variables to validate if there are better predictors than HR
fig, axes = plt.subplots(6, 3, figsize=(15, 14))
axes = axes.flatten()
# Filter out Runs, because it is just a proxy for nonHRRuns, filter out text data
predictors = [x for x in Hitters.columns if x not in ['nonHRRuns', 'Runs', 'League', 'Division', 'NewLeague']]
for i, col in enumerate(predictors):
    # Create a scatter plot with a regression line
    sns.regplot(x=col, y='nonHRRuns', data=Hitters, ax=axes[i], scatter_kws={'s': 10}) #scatter_kws to make points smaller.
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Qty of Non HR Runs")
    axes[i].set_title(f"Non HR Runs vs. {col}")
    axes[i].grid(True)
plt.tight_layout()
plt.show()


# <b>Answer:</b><p>
# At Bats and Hits actually seem like a better predictor based on the data. So let's look at them together in a model.

# In[ ]:


# Run a Ordinary Least Squares model and fit
predictors = ['AtBat', 'Hits']
X = sm.add_constant(Hitters[predictors])
y = Hitters['nonHRRuns']
model = sm.OLS(y, X)
result = model.fit(maxiter=1000)
print(result.summary())


# <b>Answer:</b><p>
# All three variables are statistically significant, so we can keep them.<p>
# What we do find, as I expected, is that the more home runs you hit, the fewer non HomeRun Runs you score. It seems paradoxical that the number is more correlated negatively, than the positive correlation between Hits and Non HomeRun Runs. There is likely some other factor at play, like the fitness of the players, possibly their speed (or lack of) running the bases.

# (d) Comment the questions (and answers) in this analysis.  Were you able to answer all of these questions?  Are all questions well defined?  Is the data good enough to answer all these?

# <b>Answer:</b><p>
# I believe I was able to answer all of my questions. My original question was possibly not as well defined as it could be, but once I visualized all of the variables together, I was able to augment with a good result.

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
print(sales.shape)
sales.head()


# In[ ]:


# Basic Sales and Cost by product
# Create subplots side by side
df_car_brand_totals = sales.groupby('product')[['sales', 'cost']].sum().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Sales Chart (Volume)
sales_plot = sns.barplot(x='product', y='sales', data=df_car_brand_totals, ax=axes[0], palette={'Bentley': 'blue', 'Tesla': 'purple'}, hue='product')
axes[0].set_title('Total Sales Volume by Product')
axes[0].set_xlabel('Product')
axes[0].set_ylabel('Sales Volume')

# Add numbers inside each bar
for p in sales_plot.patches:
    sales_plot.annotate(format(p.get_height(), ',.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                        ha='center', va='center',
                        color='white', fontsize=12, fontweight='bold')

# Format y-axis with commas
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: "{:,}".format(int(x))))

# Cost Chart (Dollars)
cost_plot = sns.barplot(x='product', y='cost', data=df_car_brand_totals, ax=axes[1], palette={'Bentley': 'blue', 'Tesla': 'purple'}, hue='product')
axes[1].set_title('Total Cost by Product')
axes[1].set_xlabel('Product')
axes[1].set_ylabel('Cost (Dollars)')

# Add numbers inside each bar
for p in cost_plot.patches:
    cost_plot.annotate(format(p.get_height(), ',.0f'),
                       (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                       ha='center', va='center',
                       color='white', fontsize=12, fontweight='bold')

# Format y-axis with commas
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: "{:,}".format(int(x))))

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plt.tight_layout()  # Adjust subplot parameters to give specified padding.
plt.show()


# In[ ]:


# Top 5 Territories
territory_totals = sales.groupby('territory_id').agg({'sales': 'sum', 'cost': 'sum'}).reset_index()
top_5_territories = territory_totals.sort_values(by='sales', ascending=False).head(5)

# Melt the DataFrame to have sales and cost in one column for plotting
df_melted = top_5_territories.melt(id_vars='territory_id', var_name='metric', value_name='total')

# Create the bar chart using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='territory_id', y='total', hue='metric', data=df_melted)
plt.title('Total Sales and Cost by Top 5 Territories')
plt.xlabel('Territory ID')
plt.ylabel('Total Units')
plt.show()


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


# Calculate the Pearson correlation coefficient between two lists.
x1 = sales['sales']
x2 = sales['cost']
# Check to make sure they are same length
if len(x1) != len(x2):
    print("Input lists must have the same length.")
    exit()

# Get the length to calculate the mean
n = len(x1)
# Get the means 
mean_x1 = sum(x1) / n
mean_x2 = sum(x2) / n

numerator = sum((x1[i] - mean_x1) * (x2[i] - mean_x2) for i in range(n))
denominator_x1 = sum((x1[i] - mean_x1) ** 2 for i in range(n))
denominator_x2 = sum((x2[i] - mean_x2) ** 2 for i in range(n))

if denominator_x1 == 0 or denominator_x2 == 0:
    pears = 0.0  # Handle division by zero

pears = numerator / (denominator_x1 ** 0.5 * denominator_x2 ** 0.5)
print(f"Pearson Correlation: {pears}")


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


# Manual calculation of Slope and Intercept
x = sales['sales']
y = sales['cost']
n = len(x)
# Get the means 
mean_x = sum(x) / n
mean_y = sum(y) / n
numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

if denominator == 0:
    slope = 0.0  # Handle division by zero
else:
    slope = numerator / denominator

intercept = mean_y - (slope * mean_x)
print(f'Intercept: {intercept:.4f}', f'Slope: {slope:.8f}')


# (d) Write down the simple linear regression model equation based on (c) results. Provide reflections on the model equation.

# <b>Answer:</b><p>
# The Simple Linear Regression Model Equation is:<p>
# $$ \hat{y} = \hat{\beta_0} + \hat{\beta_1}{x}$$ <p>
# 
# With the values calculated above:<p>
# $$ \hat{y} = {554.8872} + {0.12885535}{x}$$ <p>

# In[ ]:


# Check the equation against the actuals
costs = [692.00, 650, 600]
for x in costs:
    simple_linear_regression = intercept + (slope * x)
    print(f"If sales = {x:.2f}, Simple Linear Regression estimate of cost is : {simple_linear_regression:.2f}")
    print(f"Actual costs: {sales[sales['sales'] == x]['cost'].to_list()}")
    print('='*90)


# ### Problem 4
# 
# Points: 20
# 
# The Wisconsin Breast Cancer dataset is available as a comma-delimited text file on the UCI Machine Learning Repository {https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original}. Our goal in this problem will be to predict whether observations (i.e. tumors) are malignant or benign. We will use the original dataset in this problem. 

# (a) Obtain the data, and load it into your programming environment by pulling it directly from the web. (Do **not** download it and import it from any CSV file.) Give a brief description of the data. 

# The Wisconsin Breast Cancer dataset contains 699 samples and a total of 11 variables. Some data samples have missing data. 

# In[ ]:


import os
# Web location of data file
loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds = "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
url = loc + ds

# Load data
try:
    breast_data = pd.read_csv(url, sep=",", header=None, na_values="?")
except Exception as e:
    print(e)
    breast_data = pd.read_csv('breast_cancer.csv', index=False)
if breast_data is not None and not os.path.exists('breast_cancer.csv'):
    breast_data.to_csv('breast_cancer.csv')
breast_data.head


# In[ ]:


# Look at unique values in columns
df_breast = breast_data.copy()
columns = df_breast.columns
unique_values = {col: df_breast[col].unique() for col in columns}
for col, values in unique_values.items():
    print(f"Unique values for {col}: {values[:9]}")


# (b) Tidy the data, ensuring that each variable is properly named and cast as the correct data type. Discuss any missing data.

# In[ ]:


display(df_breast.isna().sum())
print(df_breast.dtypes)
df_breast.describe()


# <b>Answer:</b><p>
# The data from this data set is all numbers.<p>
# 0. Sample code number:            id number
# 1. Clump Thickness:               1 - 10
# 2. Uniformity of Cell Size:       1 - 10
# 3. Uniformity of Cell Shape:      1 - 10
# 4. Marginal Adhesion:             1 - 10
# 5. Single Epithelial Cell Size:   1 - 10
# 6. Bare Nuclei:                   1 - 10
# 7. Bland Chromatin:               1 - 10
# 8. Normal Nucleoli:               1 - 10
# 9. Mitoses:                       1 - 10
# 10. Class:                        (2 for benign, 4 for malignant)<p>
# There are a small number of NaNs in Column 6.

# In[ ]:


# Rename the columns
col_map_txt = """0. Sample code number:            id number
1. Clump Thickness:               1 - 10
2. Uniformity of Cell Size:       1 - 10
3. Uniformity of Cell Shape:      1 - 10
4. Marginal Adhesion:             1 - 10
5. Single Epithelial Cell Size:   1 - 10
6. Bare Nuclei:                   1 - 10
7. Bland Chromatin:               1 - 10
8. Normal Nucleoli:               1 - 10
9. Mitoses:                       1 - 10
10. Class:                        (2 for benign, 4 for malignant)"""
col_map = {}
for row in col_map_txt.split('\n'):
    col_num = int(row.split('.')[0])
    col_name = row.split('.')[1].split(':')[0].strip().lower().replace(' ', '_')
    col_map[col_num] = col_name
print(col_map)
df_breast.rename(columns=col_map, inplace=True)
df_breast.fillna(-1, inplace=True)
df_breast = df_breast.astype(int)
display(df_breast)


# (c) Split the data into a training and validation set such that a random 70\% of the observations are in the training set.

# In[ ]:


# Split the data into training and validation sets
df_train, df_val = train_test_split(df_breast, test_size=0.3, random_state=7)
display(df_train)


# (d) Fit a machine learning model to predict whether tissue samples are malignant or benign. Compute and discuss the resulting confusion matrix. Be sure to address which of the errors that are identified you consider most problematic in this context.

# In[ ]:


# Convert the malignant/benign column into 0 (benign) or 1 (malignant)
df_breast['is_malignant'] = df_breast['class'].map({2: 0, 4: 1})
df_breast


# In[ ]:


# Try a number of different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each model
conf_matrices = {}
high_model, high_acc  = '', -9999
X = df_breast.drop(['sample_code_number', 'class'], axis=1)
y = df_breast['is_malignant']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > high_acc:
        high_model, high_acc = name, accuracy
    print(f'{name} Accuracy: {accuracy}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
# Plot confusion matrices in a grid
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
axes = axes.flatten()

for ax, (name, conf_matrix) in zip(axes, conf_matrices.items()):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], ax=ax, cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# Hide any unused subplots
for i in range(len(conf_matrices), len(axes)):
    fig.delaxes(axes[i])
plt.subplots_adjust(wspace=0.5, hspace=0.9)
plt.show()
print("Highest:", high_model, high_acc)


# <b>Answer:</b><p>
# All of these models work pretty well; the accuracy changed slightly with a different random seed.<p>
# A false positive is obviously emotionally difficult, however, hopefully not life threatening. That happened most often with K-Nearest and Support Vector.<p>
# A false negative, could easily be life threatening, because the patient would think they have a benign tumor, but it is really malignant, and it could metasticize before they could get treatment. This happened most often with K-Nearest anmd Support Vector.<p>
# Decision Tree, Random Forest, Gradient and Naive Bayes all seemed to work better even with different random seeds and I would use these models to predict.

# ### Problem 5
# 
# Points: 10
# 
# Please answer the questions below by writing a short response. 

# (a) Please describe 3 **classification** machine learning methods and each model's real world application.

# 1. <b>Logistic Regression</b>. Can be used as a classification predictor as well as quantitative. It models the relationship between independent variables and the log-odds of the dependent variable to predict the probability of a binary outcome. It can be used to predict if a tumor is malignant or benign. It can be used to predict if a financial transaction is fraudulent or not.
# 2. <b>K Nearest Neighbors (KNN)</b>. Is a classification method that attempts to estimate the conditional distribution of a given X. It calculates the distances away from all elements and determines which group is most prevalent. KNN is used to create "people who bought this also bought" recommendations. By finding customers with similar purchase histories, the algorithm can suggest products that a user might like.
# 3. <b>Naive Bayes</b>. Naive Bayes is a simplified version of “Full Bayes”. It assumes that all features are conditionally independent (most of the time a faulty assumption, however the model performs well despite). It is “naive” because it does not account for order of features. Real world applications are: Spam Detection & Filtering, Email classification. Sentiment Analysis, Customer feedback analysis. Medical Diagnosis, predicting diseases or conditions based on symptoms. Fraud Detection, identify fraudulent transactions. Text Classification, automatically categorize emails or other digital content.

# In[ ]:


# Here is a simple example of Naive Bayes given their age and income 
# The model will determine if they will purchase a product

# Sample Data (replace with your own dataset)
# Example: Predicting if a person will buy a product based on age and income.
X = np.array([[25, 50000], [30, 60000], [35, 70000], [20, 40000], [40, 80000],
              [22, 45000], [38, 75000], [28, 55000], [45, 90000], [19, 38000]])
y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 0: No buy, 1: Buy

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model (Gaussian Naive Bayes for continuous features)
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Optionally visualize the confusion matrix with seaborn
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Buy', 'Wont Buy'], yticklabels=['Buy', 'Wont Buy'], ax=ax, cbar=False)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Buy', 'Wont Buy'], yticklabels=['Buy', 'Wont Buy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Example Prediction of a new data point.
new_data = np.array([[33, 65000]])
new_prediction = model.predict(new_data)
print(f"\nPrediction for new data {new_data}: {new_prediction}")


# b) This quarter, we discussed four 'Missing data' types: 1) Not missing at random (NMAR), 2) Missing completely at random(MCAR), 3) Missing at random(MAR)and 4) Missing by design (MD).  Select **2 missing data types** listed above. Describe those two missing data types meanings and provide 1 real world example for each missing data type.

# <b>Answer:</b><p>
# <b>Missing Completely At Random (MCAR)</b><p>
# Definition: The probability of a data point being missing is entirely unrelated to any observed or unobserved variables in the dataset, ie it is not related. The missing data is due to an error, completely independent of the data itself.<p>
# Examples:<p>
# Survey Data Loss: In a survey, some questionnaires are randomly lost in the mail or accidentally deleted from a database. The loss is purely random.<p>
# Technical Defects: A sensor randomly malfunctions and fails to record data points at certain intervals. .<p>
# Randomized Experiments: In a clinical trial, some participants might miss appointments due to unrelated events (e.g., car trouble, sudden illness), leading to missing data that is not related to their health or treatment.<p>
# <p>
# <b>Missing At Random (MAR)</b><p>
# Definition: The probability of a data point being missing is related to observed variables in the dataset, but not to the unobserved value itself.<p>
# Example:<p>
# Medical Records: Older patients might have more missing data in certain medical fields compared to younger patients due to having older, paper records. Within each age group, the probability of missing data is independent of the actual missing values.<p>
# Income Survey: In an income survey, women might be less likely to report their income than men. Accounting for gender, the missingness is no longer related to the income itself. The missing data is dependent on the observed gender variable.<p>
# Customer Satisfaction: Customers who purchased a specific product type might be less likely to respond to a satisfaction survey. However, once we account for the product type, the missing data is no longer related to the unobserved satisfaction level.<p>

# (c) What are the advantages and disadvantages of a very flexible (versus a less flexible) approach for regression or classification? Under what circumstances might a more flexible approach be preferred to a less flexible approach? When might a less flexible approach be preferred?

# <b>Answer:</b><p>
# In general, a more restrictive model (or less flexible) are more interpretable, while more flexible models are less interpretable. See this chart from Diez et al. 2019:<p>
# ![image.png](attachment:image.png)
# 
# Linear Regression is a fairly inflexible model because it only generates linear function such as lines. On the other hand, a linear model fit by least squares is a much more flexible model with a much more complex outcome. Many "boosted" models fit in this more flexible category.
# Source: Diez, D., Cetinkaya-Rundel, M., Barr, C., & OpenIntro. (2019, Fourth Edition). OpenIntro Statistics.

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

# <b>Answer:</b><p>
# Is this the correct formula?<p>
# $$ salary = \hat{\beta_0} + (\hat{\beta_1}*GPA) + (\hat{\beta_2}*IQ) + (\hat{\beta_3}*Sex) + (\hat{\beta_4} * GPA * IQ) + (\hat{\beta_5}* GPA * Sex) $$ <p>

# In[ ]:


# Run some sample data

# Given B values
b0, b1, b2, b3, b4, b5 = 50, 20, .07, 35, .01, -10

# GPA, IQ, Sex (1 female, 0 male)
examples = [[4.0, 120, 1], [4.0, 120, 0], [2.0, 120, 1], [2.0, 120, 0]] 
for gpa, iq, sex in examples:
    salary = b0 + (b1 * gpa) + (b2 * iq) + (b3 * sex) + (b4 * gpa * iq) + (b5 * gpa * sex)
    gender = 'female' if sex == 1 else 'male'
    print(f"GPA: {gpa}, IQ: {iq}, Sex: {gender}. Equals salary={salary}")


# <b>Answer:</b><p>
# answer `i` and `ii` are definitely not correct because we see the opposite of both of those given differing GPAs.<p>
# If we get the difference between `salaryfemale - salarymale` we should be able to find out the "break point" for GPA.<p>
# $$ salaryfemale = 50 + (20*GPA) + (.07*IQ) + (35*Sex) + (.01*GPA*IQ) + (-10*GPA*Sex) $$ <p>
# $$ salarymale = 50 + (20*GPA) + (.07*IQ) + (35*Sex) + (.01*GPA*IQ) + (-10*GPA*Sex) $$ <p>
# $$ salaryfemale - salarymale = (50 + (20*GPA) + (.07*IQ) + (35*1) + (.01*GPA*IQ) + (-10*GPA*1)) - (50 + (20*GPA) + (.07*IQ) + (35*0) + (.01*GPA*IQ) + (-10*GPA*0)) $$<p>
# $$ salaryfemale - salarymale = (50 + (20*GPA) + (.07*IQ) + 35 + (.01*GPA*IQ) + (-10*GPA)) - (50 + (20*GPA) + (.07*IQ) + (.01*GPA*IQ)) $$<p>
# $$ salaryfemale - salarymale = ((20*GPA) + (.07*IQ) + 35 + (.01*GPA*IQ) + (-10*GPA)) - ((20*GPA) + (.07*IQ) + (.01*GPA*IQ)) $$<p>
# $$ salaryfemale - salarymale = (35 + (-10*GPA))) $$<p>
# Find when the difference is positive:<p>
# $$ 35 - 10*GPA > 0 $$<p>
# $$ GPA < 3.5 $$<p>
# So when GPA is below 3.5, females earn more. Let's test.

# In[ ]:


# GPA, IQ, Sex (1 female, 0 male)
examples = [[3.6, 120, 1], [3.6, 120, 0], [3.5, 120, 1], [3.5, 120, 0], [3.4, 120, 1], [3.4, 120, 0]] 
for gpa, iq, sex in examples:
    salary = b0 + (b1 * gpa) + (b2 * iq) + (b3 * sex) + (b4 * gpa * iq) + (b5 * gpa * sex)
    gender = 'female' if sex == 1 else 'male'
    print(f"GPA: {gpa}, IQ: {iq}, Sex: {gender}. Equals salary={salary}")


# <b>Answer:</b><p>
# The correct answer is `iii. For a fixed value of IQ and GPA, males earn more on average than females provided that the GPA is high enough.`

# (b) Predict the salary of a female with IQ of 110 and a GPA of 4.0.

# In[ ]:


# Convert the equation above into Python
gpa, iq, sex = 4, 110, 1
salary = b0 + (b1 * gpa) + (b2 * iq) + (b3 * sex) + (b4 * gpa * iq) + (b5 * gpa * sex)
gender = 'female' if sex == 1 else 'male'
print(f"GPA: {gpa}, IQ: {iq}, Sex: {gender}. Equals salary={salary}")


# (c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is little evidence of an interaction effect. Justify your answer.

# <b>Answer:</b><p>
# False. It is not the `coefficient`, but rather the `P-Value`. We'll fit a model and determine the P-Value.

# In[ ]:


# Run some example data through and determine the p-value
salaries, gpa_iq = [], []
examples = [[4.0, 120, 1], [4.0, 120, 0], [2.0, 120, 1], [2.0, 120, 0], [4, 110, 1], [3.6, 120, 1], [3.6, 120, 0], [3.5, 120, 1], [3.5, 120, 0], [3.4, 120, 1], [3.4, 120, 0]] 
for gpa, iq, sex in examples:
    salary = b0 + (b1 * gpa) + (b2 * iq) + (b3 * sex) + (b4 * gpa * iq) + (b5 * gpa * sex)
    gpa_iq.append(gpa * iq)
    salaries.append(salary)
# Add more totally random values    
np.random.seed(7)
# Generate random samples
gpa = np.random.uniform(2.0, 4.0, 100)  # GPA between 2.0 and 4.0
iq = np.random.randint(90, 150, 100)    # IQ between 90 and 150
sex = np.random.randint(0, 2, 100) 
combined = np.column_stack((gpa, iq, sex))
for gpa, iq, sex in combined:
    salary = b0 + (b1 * gpa) + (b2 * iq) + (b3 * sex) + (b4 * gpa * iq) + (b5 * gpa * sex)
    gpa_iq.append(gpa * iq)
    salaries.append(salary)
    
X = sm.add_constant(gpa_iq)
y = salaries
model = sm.OLS(y, X)
result = model.fit(maxiter=1000)
print(result.summary())
print(result.pvalues)


# <b>Answer:</b><p>
# The P-Value is less than `.05` so we reject the Null Hypothosis and can state that `(GPA * IQ)` relative to predicted `salary` could not be random chance.

# ### Problem 7 - Extra Credit
# 
# Points: Max. 5
# 
# Suppose that $X_1, \ldots X_n$ form a random sample from a Poisson distribution for which the mean $\theta$ is unknown, ($\theta>0)$.<p>
# Is this MLE?<p>
# MLE stands for Maximum Likelihood Estimation.<p>
# It means you need to find the value of the parameter θ (theta) that maximizes the likelihood of observing the data you have.<p>
# Likelihood Function: You start with a probability distribution that you believe describes your data. This distribution will have one or more parameters (like θ in your problem). The likelihood function is the product of the probabilities of observing each individual data point, given a specific value for the parameter(s).<p>
# Maximization: The goal of MLE is to find the value of the parameter(s) that maximizes this likelihood function. In other words, you're finding the parameter value(s) that make the observed data most probable.

# (a) Determine the MLE of $\theta$, assuming that at least one of the observed values is different from 0. Show your work.

# <b>Answer:</b><p>
# Since $\hat\theta = r/n$<p>
# From our course materials:
# ![image.png](attachment:image.png)
#     
# where `r` is the observed number of times of "success" (ie the expected outcome) happens and `n` is the total number of times, it ends up being the observed mean.<p>
# Using a coin toss as an example, if we want to find out the liklihood of the coin toss being heads and we toss the coin 100 times and we see heads come up 47 times, the MLE of $\hat\theta$ would be `.47` or `47%`.<p>
# If you were to code the heads as `1`, and the tails as a `0`, and then take the average of all 100 coin tosses, you would also get `0.47`.

# (b) Show that the MLE of $\theta$ does not exists if every observed value is 0.

# <b>Answer:</b><p>
# If every observed value is 0, the denominator of the function is 0 and therefore the result of MLE $\theta$ is 0.

# ### Problem 8 - Extra Credit 
# 
# Points: Max. 5 
# 
# 7 Democratic party members and 7 Republican party members are running for 5 seats (1 president, 1 vice president and 3 advisors) in a committee. Voters choose 5 people at random uniformly. 
# 
# What is the probability of this event, in which Frank (Democratic) becomes the president and Will( Republican) becomes the vice-president ?

# In[ ]:


from IPython.display import Image
Image("Extra.png")


# (a) Write down your solution logics. (We want to see your reasonings: what makes the sample space (or denominator) and what makes the numerator? Why you think so?

# <b>Answer:</b><p>
# <b>It is unclear from the scenario if the votes are positional or not. Ie if a person votes for candidates `[9,  1,  6,  4, 12]` does that mean that candidate 9 is being voted for President? Or just that if candidate #9 gets the most votes, they can choose President?</b><p>
# `President` and `Vice President` also do not make sense, ie in USA politics:
# In each committee, a member of the majority party serves as its `chairperson`, while a member of the minority party serves as its `ranking member`. Four Senate committees instead refer to the ranking minority member as `vice chairperson`.<p>
# At any rate, it is obviously a very small number, let's try some tests.

# In[ ]:


# Simulation of random votes with Position 1 and 2 required
# Ie Frank must get the most votes and Will must get the second most votes
# Simulate voters
observed = 0
for i in range(1, 10000):
    np.random.seed(i)
    votes = [np.random.choice(range(1, 18), 5, replace=False) for _ in range(100)]
    # Pick a random number for Frank and Will
    # frank_dem, will_gop = np.random.choice(range(1, 15), 2)
    frank_dem, will_gop = 8, 5
    voting = {'frank_dem': 0, 'will_gop': 0}
    # Iterate over the votes
    for vote in votes:
        # If we see frank or will, increment their count
        if frank_dem in vote:
            voting['frank_dem'] += 1
        elif will_gop in vote:
            voting['will_gop'] += 1
        # Go through all the votes to tabulate the other candidates
        for v in vote:
            # Do not count Frank and Will again
            if v not in [frank_dem, will_gop]:
                v_str = str(v)
                if v_str not in voting:
                    voting[v_str] = 0
                voting[v_str] += 1
    # Sort the list to figure out the top two candidates
    sorted_list = list(dict(sorted(voting.items(), key=lambda item: item[1], reverse=True)))
    if sorted_list[0] == 'frank_dem' and sorted_list[1] == 'will_gop':
        print('Frank pres, Will vp, OBSERVED!')
        observed += 1
print(f"Ran through {i} iterations and observed {observed} favorable outcomes")


# In[ ]:


# Simulation of random votes with any Position 1-5 for Frank & Will
# Simulate voters
observed = 0
for i in range(1, 10000):
    np.random.seed(i)
    votes = [np.random.choice(range(1, 18), 5, replace=False) for _ in range(100)]
    # Pick a random number for Frank and Will
    frank_dem, will_gop = np.random.choice(range(1, 15), 2)
    voting = {'frank_dem': 0, 'will_gop': 0}
    # Iterate over the votes
    for vote in votes:
        # If we see frank or will, increment their count
        if frank_dem in vote:
            voting['frank_dem'] += 1
        elif will_gop in vote:
            voting['will_gop'] += 1
        # Go through all the votes to tabulate the other candidates
        for v in vote:
            # Do not count Frank and Will again
            if v not in [frank_dem, will_gop]:
                v_str = str(v)
                if v_str not in voting:
                    voting[v_str] = 0
                voting[v_str] += 1
    # Sort the list to figure out the top two candidates
    sorted_list = list(dict(sorted(voting.items(), key=lambda item: item[1], reverse=True)))
    if 'frank_dem' in sorted_list[:5] and 'will_gop' in sorted_list[:5]:
        observed += 1
print(f"Ran through {i} iterations and observed {observed} favorable outcomes")


# (b) Write down your calculations for the probability. [Hint: Think about whether the 2 roles (president and VP) assignment is ordered or not. Do not trust LLMs such as ChatGPT's answer completely ]

# <b>Answer:</b><p>
# If you fix Frank as President and Will as VP, the number of ways to chose 3 people out of 12 (subtract out the 2 for Frank and Will) is:<p>
# $\binom{12}{3} = \frac{12!}{3!(12-3)!} = 220$<p>
# The total number of ways to chose 5 people out of 14 is:<p>
# $\binom{14}{5} = \frac{14!}{5!(14-5)!} = 2002$<p>
# Then you divide those probabilities to get the chances that both Frank and Will are chosen:<p>
# $\text{Probability} = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} = \frac{220}{2002} \approx 0.1099$

# <b>Answer:</b><p>
# The wording is very confusing on this problem; it seems like we are supposed to answer it multiple ways. If I read this part of the question:<p>
# `Voters choose 5 people at random uniformly.`<p>
# This would indicate that it is not based on the USA Congress, in fact, I am unaware of any government that works like this unless it is a coalition government. So I will answer based on the above instructions first:

# In[ ]:


# Assuming the roles are not ordered
remain_members = math.factorial(12)/(math.factorial(3) * math.factorial(12-3))
print(f"Remaining members are voted in, posibilities: {remain_members}")
all_members = math.factorial(14)/(math.factorial(5) * math.factorial(14-5))
print(f"All 5 members are voted in, posibilities: {all_members}")
print(f"Assuming some fictional universe, the overall odds of Frank and Will be voted to the top two slots is {remain_members/all_members:.4f}")


# <b>Answer:</b><p>
# Assuming this is the USA government, the two votes are <b>actually independent of each other</b>: all the dems vote for Frank and other dems. Since the English is ambiguous, in this scenario it is still true that `Voters chose 5 people at random`, they just happen to all be in their party. Separately and independently, all the gop vote for Will and the other gop members. I will calculate this way next:

# In[ ]:


# Assuming the USA government, the two votes are actually independent
print("Frank has a 1 in 7 chance of being voted chairman.")
print("Will has a 1 in 7 chance of being voted vice chairman.")
print("You multiply these odds together to get the possibility that both things happen simultaneously:")
frank_will = (1/7) * (1/7)
print(f"Assuming the the USA congress, the overall odds of Frank and Will being voted to the top two slots is: {frank_will:.4f}")


# <b>Answer:</b><p>
# Based on my Python code in cell `123` above, the lower probability is far more likely, as I never even got one hit. 

# In[ ]:


print("Successfully executed all cells!")

