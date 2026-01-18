#!/usr/bin/env python
# coding: utf-8

# # IMT 573 - Problem Set 4 - Working With Data Part 2

# ### Instructions
# 
# Before beginning this assignment, please ensure you have access to a working instance of Jupyter Notebooks with Python 3.
# 
# 1. First, replace the “YOUR NAME HERE” text in the next cell with your own full name. Any collaborators must also be listed in this cell.
# 
# 2. Be sure to include well-documented (e.g. commented) code cells, figures, and clearly written text  explanations as necessary. Any figures should be clearly labeled and appropriately referenced within the text. Be sure that each visualization adds value to your written explanation; avoid redundancy – you do no need four different visualizations of the same pattern.
# 
# 3. Collaboration on problem sets and labs is fun, useful, and encouraged. However, each student must turn in an individual write-up in their own words as well as code/work that is their own. Regardless of whether you work with others, what you turn in must be your own work; this includes code and interpretation of results. The names of all collaborators must be listed on each assignment. Do not copy-and-paste from other students’ responses or code - your code should never be on any other student's screen or machine.
# 
# 4. All materials and resources that you use (with the exception of lecture slides) must be appropriately referenced within your assignment.
# 
# 5. Partial credit will be awarded for each question for which a serious attempt at finding an answer has been shown. Students are *strongly* encouraged to attempt each question and document their reasoning process even if they cannot find the correct answer. 
# 
# 6. After completing the assignment, ensure that your code can run from start to finish without issue. Restart the kernal and run all cells to double check.

# Name: Emma Harding
# 
# Collaborators: 

# Sources used throughout: copilot to help with debugging as suggested in class to use a LLM for debugging, it made some helpful suggestions, some not so helpful, it doesn't always get value data types correct. Is there a better LLM to use for debugging?

# For this assignment, you'll need (at least) the following packages. If the package does not load, be sure it is properly installed.

# In[4]:


get_ipython().system('pip install censusgeocode')
import censusgeocode as cg 


# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In this problem set, we will be joining disparate sets of data - namely: Seattle police crime data, information on Seattle police beats, and education attainment from the US Census. Our goal is to build a dataset where we can examine questions around crimes in Seattle and the educational attainment of people living in the areas in which the crime occurred; this requires data
# to be combined from these multiple individual sources.
# 
# As a general rule, be sure to keep copies of the original dataset(s) as you work through cleaning (remember
# data provenance!).

# ### Problem 1: Crime Data
# 
# Load the Seattle crime data from the provided `crime_data.csv` data file. The data is a modified version of the data available [here](https://data.seattle.gov/Public-Safety/Crime-Data/4fs7-3vj5). We will call this dataset the “Crime Dataset” going forward.

# In[6]:


crime_data = pd.read_csv('Crime_Data.csv')


# #### (a) Basic inspection
# 
# Perform a basic inspection of the Crime Dataset and discuss what you find. How many observations are there? Is there any anomolous or missing data?

# The are 523590 observations

# In[7]:


crime_data.tail()


# In[8]:


crime_data.describe()


# #### (b) Years of crimes
# 
# Examine the years in which crimes were committed in the dataset. What is the earliest year in the dataset? Are there any distinct trends with the annual number of crimes committed in the dataset?

# In[36]:


crime_data['Occurred Date'] = pd.to_datetime(crime_data['Occurred Date'], errors='coerce')
crime_data['Occurred Year'] = crime_data['Occurred Date'].dt.year
print(crime_data['Occurred Year'].min())
print(crime_data['Occurred Year'].max())
print(crime_data['Occurred Year'].mean())
annual_crime = crime_data.groupby('Occurred Year').size().sort_index()


# Sources: https://www.geeksforgeeks.org/python/python-working-with-date-and-time-using-pandas/ I used this source to get the pd.to_datatime
# https://docs.python.org/3/library/datetime.html
# https://pandas.pydata.org/docs/user_guide/groupby.html
# 
# I used copilot for debugging because in the beggining I was accidently returning a crime count for every single column in the crime_data dataset, copilot told me to use .size() on the end of "annual_crime = crime_data.groupby('Occurred Year')" insteadof just "annual_crime = crime_data.groupby('Occurred Year')" so my code changed to "annual_crime = crime_data.groupby('Occurred Year').size"
# 
# The earliest year in my dataset is 1908 and the latest is 2019. Over the years the crime count per year has increase dramatically. 

# #### (c) Filter data on years
# 
# Subset the data to only include crimes that were committed after 2011 (remember good practices of data
# provenance!). Going forward, we will use this data subset. Print the shape of your dataset after doing this.

# In[39]:


crime_data_filter2011 = crime_data[crime_data['Occurred Year'] > 2011]
print(crime_data_filter2011.shape)


# Sources: same as above since I used my 'Occured Year' value from my code entry above using pd.to_datetime to extract the year from ['Occurred Date']
# There are now 350,053 entries in the dataset

# #### (d) Looking at frequency of beats 
# 
# Each of the records has a police beat associated with it. You can learn more about police beats [here](https://www.seattle.gov/police/information-and-data/data/tweets-by-beat). How frequently are the beats in the (filtered) Crime Dataset listed? Are there any anomolies with how frequently some of the beats are listed? Are there missing beats?

# In[44]:


crime_beat_counts = crime_data_filter2011['Beat'].count()
print(crime_beat_counts)


# There are missing beats, 2,054 of them

# 
# 

# ### Problem 2: Police Beat Data

# Load the data on Seattle police beats provided in the police_beat_and_precinct_centerpoints.csv. This is the same data that you used in Lab 4. You can learn more about police precincts and beats [here](https://www.seattle.gov/police/about-us/about-policing/precinct-and-patrol-boundaries). The data can be found in the `Police_Beat_and_Precinct_Centerpoints.csv` file.

# In[45]:


beats_data = pd.read_csv('Police_Beat_and_Precinct_Centerpoints.csv')


# #### (a) Missing beats
# 
# Does the (filtered) Crime Dataset include police beats that are not present in the Beats Dataset? If so, how many and
# with what frequency do they occur? Would you say that these comprise a large number of the observations
# in the Crime Dataset or are they rather infrequent? Do you think removing them would drastically alter the
# scope of the Crime Dataset?

# In[50]:


crime_data_filter2011['Beat'].unique()


# In[49]:


crime_data_filter2011['Beat'].nunique()


# In[53]:


beats_data['Name'].unique


# In[55]:


beats_data['Name'].nunique()


# There are 57 unique beat occurances in the beats_data but there are 59 in my filtered crime dataset

# In[61]:


crime_beat_counts = crime_data_filter2011['Beat'].value_counts()


# In[70]:


crime_beat_counts = crime_data_filter2011['Beat'].value_counts()
official_beats = set(beats_data['Name'].dropna().unique())
crime_beats = set(crime_beat_counts.index)
beats_count = beats_data['Name'].value_counts()
unmatched_beats = crime_beats - official_beats
unmatched_table = crime_beat_counts.loc[list(unmatched_beats)].reset_index()
unmatched_table.columns = ['Beat', 'Frequency']
print(unmatched_table)


# In[75]:


beat_beat_counts = beats_data['Name'].value_counts()
unmatched_beats2 = official_beats - crime_beats
unmatched_table = beat_beat_counts.loc[list(unmatched_beats2)].reset_index()
unmatched_table.columns = ['Beat', 'Frequency']
print(unmatched_table)


# Sources: https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
# https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
# 
# There are 3 values in the beats data that don't appear in the crimes data and 6 values in the crime data that don't appear in the beats data

# 
# 

# #### (b) Filtering beats
# 
# Let’s remove all instances in the (filtered) Crime Dataset that have beats which occur fewer than 10 times the dataset. Also remove any observations with missing beats. After only keeping years of interest and filtering based on frequency of the beat, how many observations do we now have in the Crime Dataset?

# In[83]:


crime_clean_2011 = crime_data_filter2011.dropna(subset=['Beat'])
beat_counts_2011 = crime_clean_2011['Beat'].value_counts()
morethen10_beats2011 = beat_counts_2011[beat_counts_2011 >= 10].index
morethen10_crime_filtered_2011 = crime_clean_2011[crime_clean_2011['Beat'].isin(morethen10_beats2011)]
print(len(morethen10_crime_filtered_2011))


# Sources: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
# 
# I was having a lot of issues with this one so I used copilot to suggest fixes for my code because I was trying to use .dropna to filter a DataFrame directly with a numeric condition when .dropna only works on Series like value_counts()
# 
# There are 347,980 overservations in the dataset after dropping empty values and only using values with more then 10 beat number occurances

# #### (c) Filtering beats
# 
# To join the Beat Dataset to census data, we must have census tract information. In a previous lab, you used an API to get the census information for each police beat. Use the code from that lab to add a column named "tract" to the Beats Dataset. Remember good practices of data provenance!

# In[92]:


def get_census_tract (longitude, latitude):
    result = cg.coordinates(x=longitude, y=latitude)
    result.input
    geoid = result['Census Tracts'][0]['GEOID']
    return (f"{geoid}")


beats_data['tract'] = beats_data.apply(lambda row: get_census_tract(row['Longitude'], row['Latitude']), axis=1)


# In[93]:


print(beats_data['tract'])


# In[ ]:


Sources:
https://www.geeksforgeeks.org/pandas/adding-new-column-to-existing-dataframe-in-pandas/
https://www.geeksforgeeks.org/python/applying-lambda-functions-to-pandas-dataframe/

Copilot suggested using lambdas to define a function without having to write a full function to add the tract column to my beats_data dataset.


# #### (d) Process for joining
# 
# We will eventually join the Beats Dataset to the Crime Dataset. We could have joined the two and then
# found the census tracts for each beat. Would there have been a particular advantage/disadvantage to doing this join first and then finding census tracts? If so, what is it? (NOTE: you do not need to write any code to answer this)

# Some disadvantages are that it would be harder to check for missing values first so it's best to find census tracts and then join the datasets together

# 
# 

# #### (e) Extracting 11-digit codes 
# The census data uses an 11-digit code that consists of the state, county, and tract code. It does not include the block code. To join the census data to the Beats Dataset, we must have this code for each of the beats. Extract the 11-digit code for each of the beats in the Beats Dataset. The 11 digits consist of the 2 state digits, 3 county digits, and 6 tract digits. Add a column to the (modified) Beats Dataset named `census_id` with the 11-digit code for each beat. Ensure that the values in the `census_id` column are stored as `int64`.
# 
# You can learn more about the codes [here](https://transition.fcc.gov/form477/Geo/more_about_census_blocks.pdf). 

# In[96]:


beats_data['tract'] = beats_data['tract'].astype(str)
beats_data['census_id'] = beats_data['tract'].str[:11]
beats_data['census_id'] = beats_data['census_id'].astype('int64')


# In[ ]:


sources: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.astype.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html



# #### (f) Extracting 11-digit codes from census 
# 
# Now, we will examine Census Dataset in the `census_data_2020_edu_attainment.csv` file. The data includes counts of education attainment across different census tracts. Note how this data is in a "wide" format and how it can be converted to a "long" format. For now, we will work with it as is. 
# 
# The census data contains a `GEO_ID` column. Among other things, this variable encodes the 11-digit code that we had extracted above for each of the police beats. Specifically, when we look at the characters after the characters "US" for values of GEO.id, we see encodings for state, county, and tract, which should align with the beats we had above. Extract the 11-digit code from the `GEO_ID` column. Add a column named `census_id` to the Census Dataset with the 11-digit code for each census observation. Be sure to practice good data provenance when modifying a dataset.

# In[107]:


census_df = pd.read_csv("census_data_2020_edu_attainment.csv")
census_df['census_id'] = census_df['GEO_ID'].str.extract(r'US(\d{11})')[0]
census_df['census_id'] = census_df['census_id'].astype('int64')
census_df.to_csv("census_data_with_census_id.csv", index=False)


# Sources:
# sources: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.astype.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
# 
# I had to use copilot to figure out ".str.extract(r'US(\d{11})')[0]", I was struggling with the exact correct values to put in place to get an error free outcome

# ### Problem 3: Census Data
# 
# #### (a) Join datasets part 1
# 
# Join the Census data in the with the (modified) Beat Dataset using the 11-digit codes as keys. Be sure that you do not lose any of the police beats when doing this join (i.e. your output dataframe should have the same number of rows as the cleaned Beats Dataset - use the correct join). Are there any police beats that do not have any associated census data? If so, how many? Save the resulting data in a different variable (i.e. don't overwrite the Beats or Census Datasets).

# In[108]:


beats_census_joined = beats_data.merge(census_df, on='census_id', how='left')
original_beat_count = len(beats_data)
joined_beat_count = len(beats_census_joined)
missing_census = beats_census_joined['GEO_ID'].isna().sum()
print(original_beat_count)
print(joined_beat_count)
print(missing_census)


# Sources: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isna.html
# 
# There are no police beats without associated census date.
# 
#     I used copilot for suggestions on how to figure out this problem because I was having trouble and it told me to use a left join
#     I used a left join because left joins return all the records from the first DF regardless of if the keys in the first DF are in the the second DF
#   

# #### (b) Join datasets part 2
# 
# Then, join the Crime Dataset to our joined beat/census data. We can do this using the police beat name. Again, be sure you do not lose any observations from the (filtered) Crime Dataset. What are the final dimensions of the joined dataset? Save this joined dataset as you'll use it in a future problem set.

# In[102]:


final_joined_df = crime_filtered_2011.merge(beats_census_joined, left_on='Beat', right_on='Name', how='left')
print(f"Final dataset shape: {final_joined_df.shape}")


# Sources: https://www.geeksforgeeks.org/python/different-types-of-joins-in-pandas/
# The final dimensions of the dataset are 347,980 rows and 34 columns

# In[ ]:





# In[ ]:




