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

stratus_data_cleaned = stratus_data.replace([np.inf, -np.inf], np.nan).dropna()
HHH_data_cleaned = competitor1_data.replace([np.inf, -np.inf], np.nan).dropna()
JOE_data_cleaned = competitor2_data.replace([np.inf, -np.inf], np.nan).dropna()

def fit_ar_model(data):
    # Create the lagged_return column within the provided data DataFrame
    data_copy = data.copy()
    data_copy['lagged_return'] = data_copy['simple_return'].shift(1)
    # Fit the AR(1) model
    model = sm.OLS(data_copy['simple_return'][1:], sm.add_constant(data_copy['lagged_return'][1:]))
    results = model.fit()
    return results

# Fit AR(1) model for each stock
stratus_ar_results = fit_ar_model(stratus_data_cleaned)
competitor1_ar_results = fit_ar_model(HHH_data_cleaned)
competitor2_ar_results = fit_ar_model(JOE_data_cleaned)

# Display results
print(stratus_ar_results.summary())
#print(competitor1_ar_results.summary())
#print(competitor2_ar_results.summary())

#2)
# After calculating returns
stratus_data_cleaned['DayOfWeek'] = stratus_data_cleaned.index.dayofweek
day_dummies = pd.get_dummies(stratus_data_cleaned['DayOfWeek'])

# Regression for Day of the Week Effect
Y = stratus_data_cleaned['simple_return']
X = day_dummies
model = sm.OLS(Y, X).fit()
print(model.summary())

# E autocorrelation test:

acf_values = acf(stratus_data_cleaned['simple_return'].dropna())
pacf_values = pacf(stratus_data_cleaned['simple_return'].dropna())

# Plot ACF and PACF
plot_acf(stratus_data_cleaned['simple_return'].dropna())
plot_pacf(stratus_data_cleaned['simple_return'].dropna())
#share price 
plot_acf(stratus_data_cleaned['Adj Close'].dropna())
plot_pacf(stratus_data_cleaned['Adj Close'].dropna())

# AR(1) without constant
# Remove NaNs and infinite values from the dataset


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
day_dummies_cleaned.columns = ['D1', 'D2', 'D3', 'D4', 'D5'] 

# 4. OLS Regression for Day-of-the-Week Effect 
X_cleaned = day_dummies_cleaned
Y_cleaned = stratus_data_cleaned['simple_return']
model_day_effect_cleaned = sm.OLS(Y_cleaned, X_cleaned).fit()
print(model_day_effect_cleaned.summary())


#residual checks
plt.clf()
predicted_values = model_day_effect_cleaned.predict(X_cleaned)
residuals = Y_cleaned - predicted_values
mean_residual = residuals.mean()
print("Mean of Residuals:", mean_residual)
plt.scatter(predicted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')



for column in X_cleaned.columns:
    corr = residuals.corr(X_cleaned[column])
    print(f"Correlation between residuals and {column}: {corr}")



import scipy.stats as stats

# Histogram of Residuals
plt.hist(residuals, bins=30, edgecolor='black')
plt.title('Histogram of Residuals')
#plt.show()

# Normality Test (e.g., Shapiro-Wilk)
shapiro_test = stats.shapiro(residuals)
print("Shapiro-Wilk Test:", shapiro_test)


from statsmodels.stats.outliers_influence import variance_inflation_factor


# Assuming X_cleaned is a DataFrame of your independent variables
correlation_matrix = X_cleaned.corr()
print(correlation_matrix)

# Assuming X_cleaned is your DataFrame of independent variables
vif_data = pd.DataFrame()
vif_data["feature"] = X_cleaned.columns
vif_data["VIF"] = [variance_inflation_factor(X_cleaned.values, i) for i in range(len(X_cleaned.columns))]
print(vif_data)
print(X_cleaned.head())
