import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf


stratus = 'STRS' 
competitor = ['HHH','JOE']

start_date  = '2004-01-02'
end_date = '2024-01-02'

stratus_data = yf.download(stratus, start=start_date, end=end_date)
competitor1_data = yf.download(competitor[0], start=start_date, end=end_date)
competitor2_data = yf.download(competitor[1], start=start_date, end=end_date)

def cal_return(data):
    data['simple_return'] = data['Adj Close'].pct_change()*100
    return data

stratus_data = cal_return(stratus_data)
competitor1_data = cal_return(competitor1_data)
competitor2_data = cal_return(competitor2_data)

#Descriptive Statistics and return distributions

# Calculate summary statistics
stratus_summary = stratus_data['simple_return'].describe()
competitor1_summary = competitor1_data['simple_return'].describe()
competitor2_summary = competitor2_data['simple_return'].describe()

# Calculate skewness and kurtosis
stratus_skew_kurt = {'Skewness': stratus_data['simple_return'].skew(), 'Kurtosis': stratus_data['simple_return'].kurt()}
competitor1_skew_kurt = {'Skewness': competitor1_data['simple_return'].skew(), 'Kurtosis': competitor1_data['simple_return'].kurt()}
competitor2_skew_kurt = {'Skewness': competitor2_data['simple_return'].skew(), 'Kurtosis': competitor2_data['simple_return'].kurt()}

# Time Series Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

ax1.plot(stratus_data['simple_return'], label='STRS')
ax1.set_title('Time Series of Returns for STRS')
ax1.legend()

ax2.plot(competitor1_data['simple_return'], label='HHH')
ax2.set_title('Time Series of Returns for HHH')
ax2.legend()

ax3.plot(competitor2_data['simple_return'], label='JOE')
ax3.set_title('Time Series of Returns for JOE')
ax3.legend()



# Histogram of Return Distributions
plt.figure(figsize=(12, 6))
sns.histplot(stratus_data['simple_return'], bins=30, label='Stratus', kde=True)
sns.histplot(competitor1_data['simple_return'], bins=30, label='Competitor 1', kde=True)
sns.histplot(competitor2_data['simple_return'], bins=30, label='Competitor 2', kde=True)
plt.title('Return Distribution Histogram')
plt.legend()


def fit_ar_model(data):
    # Create the lagged_return column within the provided data DataFrame
    data['lagged_return'] = data['simple_return'].shift(1)

    # Check for missing and infinite values in lagged_return
    missing_data = data['lagged_return'].isna()
    infinite_data = np.isinf(data['lagged_return'])

    # Filter out rows with missing or infinite values
    data = data[~missing_data & ~infinite_data]

    # Fit the AR(1) model
    model = sm.OLS(data['simple_return'][1:], sm.add_constant(data['lagged_return'][1:]))
    results = model.fit()
    return results

# Fit AR(1) model for each stock
stratus_ar_results = fit_ar_model(stratus_data)
competitor1_ar_results = fit_ar_model(competitor1_data)
competitor2_ar_results = fit_ar_model(competitor2_data)

# Display results
print(stratus_ar_results.summary())
print(competitor1_ar_results.summary())
print(competitor2_ar_results.summary())


# After calculating returns
stratus_data['DayOfWeek'] = stratus_data.index.dayofweek
day_dummies = pd.get_dummies(stratus_data['DayOfWeek']).iloc[:, 0:4]  # Exclude one day to avoid multicollinearity

# Regression for Day of the Week Effect
Y = stratus_data['simple_return']
X = day_dummies
model = sm.OLS(Y, X).fit()
print(model.summary())

# E autocorrelation test:

acf_values = acf(stratus_data['simple_return'].dropna())
pacf_values = pacf(stratus_data['simple_return'].dropna())

# Plot ACF and PACF
plot_acf(stratus_data['simple_return'].dropna())
plot_pacf(stratus_data['simple_return'].dropna())
#share price 
plot_acf(stratus_data['Adj Close'].dropna())
plot_pacf(stratus_data['Adj Close'].dropna())

# AR(1) without constant
# Remove NaNs and infinite values from the dataset
stratus_data_cleaned = stratus_data.replace([np.inf, -np.inf], np.nan).dropna()

# Then proceed with fitting the AR(1) model without a constant
model_no_const = sm.OLS(stratus_data_cleaned['simple_return'][1:], stratus_data_cleaned['simple_return'].shift(1)[1:]).fit()
print(model_no_const.summary())


# AR(1) with constant
model_with_const = sm.OLS(stratus_data_cleaned['simple_return'][1:], sm.add_constant(stratus_data_cleaned['simple_return'].shift(1)[1:])).fit()
print(model_with_const.summary())


plt.scatter(stratus_data_cleaned['simple_return'].shift(1), stratus_data_cleaned['simple_return'])
plt.xlabel('Lagged Return')
plt.ylabel('Current Return')
plt.title('Scatter Plot of Lagged vs Current Returns')

plt.figure()
plt.plot(stratus_data_cleaned['simple_return'])
plt.title('Time Series of STRS Returns')


#Detect Seasonality in Financial Markets
stratus_data_cleaned['DayOfWeek'] = stratus_data_cleaned.index.dayofweek

# 3. Create Dummy Variables for the cleaned data
day_dummies_cleaned = pd.get_dummies(stratus_data_cleaned['DayOfWeek'])
day_dummies_cleaned.columns = ['D1', 'D2', 'D3', 'D4', 'D5']  # Rename columns for clarity

# 4. OLS Regression for Day-of-the-Week Effect 
X_cleaned = day_dummies_cleaned
Y_cleaned = stratus_data_cleaned['simple_return']
model_day_effect_cleaned = sm.OLS(Y_cleaned, X_cleaned).fit()
print(model_day_effect_cleaned.summary())



