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


# Basic descriptive analysis of data
columns = affairs.columns
unique_values = {col: affairs[col].unique() for col in columns}
for col, values in unique_values.items():
    print(f"Unique values for {col}: {values[:9]}")

# In[7]:


# Answer to Question a
# Descriptive information about the Dataset
print("Answer to Question 1a Descriptive information about the Dataset")
print(affairs.isna().sum())
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
print(f"Proportion of families who do not have any children: {n_have_no_child / n_total_respondents:.2%}")

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
print(df_affairs)

# In[12]:


# Since we encountered divide by zero, let's take log + 1
df_affairs['log_affairs'] = df_affairs['affairs'].apply(np.log1p)
affairs['log_affairs'] = affairs['affairs'].apply(np.log1p)
print(df_affairs)

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
        5: 'managerial, administrative, business',  # Keep
        6: 'professional'  # Keep
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

print(df_affairs.head())
df_affairs_with_dummies = create_dummies(df_affairs, ['educ', 'occupation', 'occupation_husb'])
print(df_affairs_with_dummies.head())

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
print(df_means)

for col in df_affairs_means.columns:
    if col not in ['rate_marriage', 'affairs', 'log_affairs', 'had_affair']:
        df_affairs_means.loc[:, col] = df_means[col]
# Add back in the non-simulated data (ie real)
additional_cols = ['had_affair', 'rate_marriage']
for col in additional_cols:
    df_affairs_means.loc[:, col] = df_affairs[col].values
print(df_affairs_means)

# In[51]:


# Predict using the Best Model
simulated_predictions = best_fit_model_e.predict(df_affairs_means)
df_affairs_means_predicted = df_affairs_means.copy()
df_affairs_means_predicted['predicted_affair'] = simulated_predictions.apply(lambda x: '{:.4f}'.format(x))
print(df_affairs_means_predicted)
