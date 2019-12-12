# Data Visualiztion
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Data Manipulation
import pandas as pd
import numpy as np

# Directory changes
import os

# Statistical tests
from scipy import stats
import statistics
import scipy


os.chdir('C:\\Users\\...\\Downloads\\us-household-income-stats-geo-locations')

for file in os.listdir(os.getcwd()):
    if '.csv' in file:
        chunks = [chunk for chunk in pd.read_csv(os.getcwd() + '\\{}'.format(file), chunksize = 2500, encoding = 'latin-1')]
        df = pd.concat(chunks, ignore_index = True)

for state in set(df['State_ab'].to_list()):
    for val in df[df['State_ab'].str.match(state)]['County'].value_counts().reset_index()['County'].to_list():
        if val > 10:
            print(state, val)
# It appears the data seems to have a very high bias regarding the amount of cities that were 'polled' for a 
# given State's county... 
# That is, I find it misleading to draw conclusion's regarding a state's behavior as most of that state's behavior is
# dictated by mostly one of the State's counties... 
# For example, in Illinois, Adams County has the most cities in this dataset. Therefore, it may be misleading
# to generalize Illinois' Average Household income as we do not have many (only 19) observations in Chicago's county.
# Nonetheless, we will choose smaller random sample sizes to minimize the amount of undesired influence. 
# We will use a Z Test for larger sample sizes (due to the Central Limit theorem) for n >= 30.
# We will use a T Test for smaller smaple sizes for n < 30.
# This will help us possibly minimize the amount of undesired influence.

# We write a function that will permit us to visualize the distribution of the Data before
# after a Log Transformation...
def view_data(state):
    # Get data...
    mvals = df[(df['State_Name'].str.match(state)) & (df['Mean'] != 0) ]['Mean']
    
    # Print Histogram...
    fig = go.Figure(data=[go.Histogram(x=mvals)])

    fig.update_layout(
        title_text='{} Mean Household Income'.format(state), 
        xaxis_title_text='Household Income', 
        yaxis_title_text='Count', 
        bargap=0.2, 
        bargroupgap=0.1 
        )

    fig.show()
    
    # Print Probability Plot...
    stats.probplot(mvals, plot=plt)
    plt.show()
    
    # Log Transformation...
    mvals_log = df[(df['State_Name'].str.contains(state)) & (df['Mean'] != 0)]['Mean'].apply(lambda x: np.log(x))
    
    # Log Transformation Histogram...
    fig = go.Figure(data=[go.Histogram(x=mvals_log)])

    fig.update_layout(
        title_text='{} Mean Household Income (Log Transformation)'.format(state), 
        xaxis_title_text='Household Income',
        yaxis_title_text='Count', 
        bargap=0.2, 
        bargroupgap=0.1 
    )

    fig.show()
    
    # Log Transforamtion Probability Plot
    stats.probplot(mvals_log, plot=plt)
    plt.show()
    
# We begin our Hypothesis Tests...
# We write a function that will complete a Z-Test on Two States and a specified sample size...
def z_test(states, sample_size):
    values = []
    for state in states:
        data = df[(df['State_Name'].str.match(state)) & (df['Mean'] != 0)]['Mean'].apply(lambda x: np.log(x)).sample(n=sample_size).to_list()
        st_dev = statistics.stdev(data)
        mean = statistics.mean(data)
        values.append([mean, st_dev])

    z_score = (values[0][0] - values[1][0])/((values[0][1]**2 + values[1][1]**2)/sample_size)**.5

    return float(scipy.stats.norm.sf(abs(z_score))*2)

# Let's use the Z-Test on any two states and write this as a function...
# In orde to do so, we operate under the assumption that the state in the 'state' list is more 'normal'
# with a Log Transformation...
# In addition, the states variable must contain a list of states as a string.
# sample_size must be an int.
# loop must be an int.
def loop_ztest(states, sample_size, loop):
    reject = 0
    not_reject = 0

    for val in range(0,loop):
        p_value = z_test(states, sample_size)
        if p_value < 0.05:
            print(val, p_value, 'Reject H0')
            reject += 1
        
        else:
            print('Do not reject H0...')
            not_reject += 1
    return print(reject, not_reject)

loop_ztest(['Michigan', 'California'], 50, 50)