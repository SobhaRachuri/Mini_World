#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA EXPLORATION
# IMPORTING LIBRARIES FOR DATA ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#IGNORE HARMLESS WARNINGS
import warnings
warnings.filterwarnings("ignore")
#SET TO DISPALY ALL COLUMNS IN THE DATA SET
pd.set_option("display.max_columns",None)


# In[2]:


#LOADING THE DATA SET 
sddata=pd.read_excel(r"C:\Users\R Sobha Supriya\Desktop\Mini World New2.xlsx",header=0)
#COPY THE BACK-UP FILES
sddata_bk=sddata.copy()
#DISPLAY THE DATA
sddata


# In[3]:


#DISPLAY THE DATA SET INFORMATION
sddata.info()


# In[4]:


#DISPLAY THE TOTAL SHAPE THAT INCLUDES NO OF RECORDS AND ATTRIBUTES OF DATA SET
sddata.shape


# In[5]:


#TO DISPLAY MISSING VALUES IN DATA SET
sddata.isnull().sum()


# In[6]:


#4
#IDENTIFY DUPLICATES IN DATA SET
sddata_dup=sddata[sddata.duplicated(keep='last')]
#DISPLAY DUPLICATE RECORDS
sddata_dup


# In[7]:


#TO IDENTIFY DUPLICATES IN DATA SET
sddata.duplicated().any()


# In[8]:


sddata.nunique()


# In[9]:


#cols=['Category','Region','QuantitySold','Date']


# In[10]:


del sddata['Transaction ID']


# In[11]:


del sddata['Customer ID']


# In[12]:


del sddata['Product ID']


# In[13]:


sddata.info()


# In[14]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Print the column names to verify
print(sddata.columns)

# Display the first few rows of the DataFrame
print(sddata)


# In[15]:


sddata.columns = sddata.columns.str.strip()


# In[16]:


sddata.columns


# In[17]:


sddata.info()


# In[18]:


sddata


# In[19]:


sddata['Region'] = sddata['Region'].str.replace(' ', '')
sddata


# In[20]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Remove spaces from values in the "Category" column
sddata['Category'] = sddata['Category'].str.replace(' ', '')

# Now you can group and access the columns correctly
category_sales = sddata.groupby('Category')['SalesAmount'].sum()

# Print the results
print("Total Sales Amount per Category:")
print(category_sales)


# In[21]:


sddata


# In[22]:


sddata['Category'].value_counts


# In[23]:


#use label encoder to handle categorical data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
sddata["Category"]=LE.fit_transform(sddata[["Category"]])


# In[24]:


sddata


# In[25]:


sddata['Region'].value_counts


# In[26]:


sddata.info()


# In[27]:


#use label encoder to handle categorical data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
sddata["Region"]=LE.fit_transform(sddata[["Region"]])


# In[28]:


sddata.info()


# In[29]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Specify the column containing the numerical data you want to analyze (e.g., "Sales Amount ($)")
numerical_column = "SalesAmount"

# Calculate the interquartile range (IQR)
Q1 = sddata[numerical_column].quantile(0.25)
Q3 = sddata[numerical_column].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = sddata[(sddata[numerical_column] < lower_bound) | (sddata[numerical_column] > upper_bound)]

# Print the outliers
print("Outliers:")
print(outliers)


# In[30]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Specify the column containing the numerical data you want to analyze (e.g., "Sales Amount ($)")
numerical_column = "QuantitySold"

# Calculate the interquartile range (IQR)
Q1 = sddata[numerical_column].quantile(0.25)
Q3 = sddata[numerical_column].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = sddata[(sddata[numerical_column] < lower_bound) | (sddata[numerical_column] > upper_bound)]

# Print the outliers
print("Outliers:")
print(outliers)


# In[31]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Calculate the mean of the "Sales Amount ($)" column
sales_amount_mean = sddata['SalesAmount'].mean()
sales_total=sddata["SalesAmount"].sum()

# Print the mean value
print("Mean Sales Amount: ${:.2f}".format(sales_amount_mean))
print("total Sales Amount: ${:.2f}".format(sales_total))


# In[32]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Remove spaces from values in the "Category" column
#sddata['Category'] = sddata['Category'].str.replace(' ', '')

# Now you can group and access the columns correctly
category_sales = sddata.groupby('Category')['SalesAmount'].sum()

# Print the results
print("Total Sales Amount per Category:")
print(category_sales)


# In[33]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Assuming you have an 'EncodedCategory' column containing numerical category codes
max_encoded_category = sddata['Category'].max()

# Print the maximum encoded category
print("Maximum Encoded Category:", max_encoded_category)


# In[34]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Convert the "Region" column to string type
sddata['Region'] = sddata['Region'].astype(str)

# Remove spaces from values in the "Region" column
sddata['Region'] = sddata['Region'].str.replace(' ', '')

# Print the updated DataFrame
print(sddata)


# In[35]:


#BY LABEL ENCODER EAST-0,WEST-3,NORTH-1,SOUTH-2 CAN BE ASSIGNED


# In[36]:


import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Remove spaces from values in the "Category" column
#sddata['Region'] = sddata['Region'].str.replace(' ', '')

# Now you can group and access the columns correctly
category_sales = sddata.groupby('Region')['SalesAmount'].sum()

# Print the results
print("Total Sales Amount per Category:")
print(category_sales)


# In[37]:


#MAX SALES RECORDED IN REGION OF NORTH 


# In[38]:


#CORRELATION BETWEEN CATEGORY AND QUANTITY SOLD
#import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Calculate the correlation between numerical 'Region' and 'SalesAmount'
correlation = sddata['Category'].corr(sddata['QuantitySold'])

# Print the correlation coefficient
print("Correlation between CATEGORY and  SALESAMOUNT:", correlation)


# In[39]:


#WEAK NEGATIVE CORELATION BETWEEN CATEGORY AND QUANTITY SOLD


# In[40]:


#CORRELATION BETWEEN CATEGORY AND SALES AMOUNT
#import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Calculate the correlation between numerical 'Region' and 'SalesAmount'
correlation = sddata['Category'].corr(sddata['SalesAmount'])

# Print the correlation coefficient
print("Correlation between CATEGORY and  SALESAMOUNT:", correlation)


# In[41]:


#MODERATE POSITIVE CORELATION BETWEEN CATEGORY AND SALES AMOUNT THAT MEANS IF VALUES IN CATEGORY DIFFERS THEN
#SALES AMOUNT ALSO INCREASES .


# In[42]:


#CORRELATION BETWEEN QUANTITY SOLD AND SALES AMOUNT
#import pandas as pd

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Calculate the correlation between numerical 'Region' and 'SalesAmount'
correlation = sddata['QuantitySold'].corr(sddata['SalesAmount'])

# Print the correlation coefficient
print("Correlation between CATEGORY and  SALESAMOUNT:", correlation)


# In[43]:


#TRADE OFF RELATION SHIP BETWEEN QUANTITY SOLD AND SALES AMOUNT
#It's possible that products with lower prices (higher quantities sold) contribute to lower individual sales amounts.


# In[44]:


# INDICATES RELEATION BETWEEN QUANTITY SOLD AND SALES AMOUNT
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(sddata['QuantitySold'], sddata['SalesAmount'], alpha=0.5)
plt.title('Scatter Plot of Quantity Sold vs. Sales Amount')
plt.xlabel('Quantity Sold')
plt.ylabel('Sales Amount')
plt.grid(True)
plt.show()


# In[45]:


#BAR PLOT FOR CATEGORY AND SALES AMOUNT
import matplotlib.pyplot as plt

category_sales = sddata.groupby('Category')['SalesAmount'].sum()

plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar')
plt.title('Total Sales Amount by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





# In[46]:


#BOX PLOT FOR SALES AMOUNT BY CATEGORY
plt.figure(figsize=(10, 6))
sddata.boxplot(column='SalesAmount', by='Category', vert=False)
plt.title('Box Plot of Sales Amount by Category')
plt.xlabel('Sales Amount')
plt.ylabel('Category')
plt.tight_layout()
plt.show()


# In[47]:


#PAIR PLOT  FOR SALES AMOUNT,QUANTITY SOLD,CATEGORY 
sns.pairplot(sddata, vars=['SalesAmount', 'QuantitySold', 'Category'])
plt.suptitle('Pair Plot of Sales Amount, Quantity Sold, and Category')
plt.tight_layout()
plt.show()


# In[48]:


#CORELATION HEAT MAP FOR SALES AMOUNT QUANTITY SOLD
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = sddata[['SalesAmount', 'QuantitySold']].corr()

# Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Sales Amount vs. Quantity Sold')
plt.tight_layout()
plt.show()


# In[49]:


#identify the independent and Target (dependent) variables
IndepepVar=[]
for col in sddata.columns:
    if col!="SalesAmount":
        IndepepVar.append(col)
TargetVar="SalesAmount" 
x=sddata[IndepepVar]
y=sddata[TargetVar]


# In[50]:


# Split the data into train and test (random sampling)

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Display the shape for train & test data

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[51]:


sddata.dtypes


# In[52]:


#del sddata['Date']


# In[53]:


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the 'Region' column
x['Region'] = label_encoder.fit_transform(x['Region'])


# In[54]:


#compare classification algorithms
#load the results
EM_Results=pd.read_csv(r"C:\Users\R Sobha Supriya\Desktop\internship datsets&files\EMResults.csv",header=0)
#display first 5 records
EM_Results.head()


# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data into a pandas DataFrame (replace 'data.csv' with your file)
# df = pd.read_csv('data.csv')

# Assuming 'x' contains your features and 'y' is your target variable
x = sddata.drop(['SalesAmount', 'Date'], axis=1)  # Remove 'Date' and 'SalesAmount'
y = sddata['SalesAmount']

# Convert 'Region' to integer using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x['Region'] = label_encoder.fit_transform(x['Region'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=14)

# Create the Linear Regression model
model_lr = LinearRegression()

# Fit the model to the training data
model_lr.fit(x_train, y_train)

# Predict on the testing data
y_pred = model_lr.predict(x_test)

# Calculate mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[56]:


#multiple regression
# Build the multi regression model

from sklearn.linear_model import LinearRegression  

# Create object for the model

ModelMLR = LinearRegression()

# Train the model with training data

ModelMLR.fit(x_train, y_train)

# Predict the model with test dataset

y_pred = ModelMLR.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))  
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))  
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))
print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y_pred)*100,3), '%')
# Define the function to calculate the MAPE - Mean Absolute Percentage Error

def MAPE (y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Evaluation of MAPE
result = MAPE(y_test, y_pred)
print('Mean Absolute Percentage Error (MAPE):', round(result, 3), '%')

# Calculate Adjusted R squared values 

r_squared = round(metrics.r2_score(y_test, y_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
print('Adj R Square: ', adjusted_r_squared)


# In[57]:


#compare classification algorithms
#load the results
RGRResults=pd.read_csv(r"C:\Users\R Sobha Supriya\Desktop\internship datsets&files\RGRResults.csv",header=0)
#display first 5 records
RGRResults.head()


# In[58]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the Decision Tree Regressor model
model_dtr = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
model_dtr.fit(x_train, y_train)

# Predict on the testing data
y_pred_dtr = model_dtr.predict(x_test)

# Calculate evaluation metrics
mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
mse_dtr = mean_squared_error(y_test, y_pred_dtr)
rmse_dtr = np.sqrt(mse_dtr)
r2_dtr = r2_score(y_test, y_pred_dtr)

print("Decision Tree Regressor:")
print("Mean Absolute Error:", mae_dtr)
print("Mean Squared Error:", mse_dtr)
print("Root Mean Squared Error:", rmse_dtr)
print("R-squared:", r2_dtr)


# In[59]:


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the Random Forest Regressor model
model_rf = RandomForestRegressor(random_state=42)

# Fit the model to the training data
model_rf.fit(x_train, y_train)

# Predict on the testing data
y_pred_rf = model_rf.predict(x_test)

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor:")
print("Mean Absolute Error:", round(mae_rf, 3))
print("Mean Squared Error:", round(mse_rf, 3))
print("Root Mean Squared Error:", round(rmse_rf, 3))
print("R-squared:", round(r2_rf, 6))



# In[60]:


# Create the Extra Trees Regressor model
model_et = ExtraTreesRegressor(random_state=42)

# Fit the model to the training data
model_et.fit(x_train, y_train)

# Predict on the testing data
y_pred_et = model_et.predict(x_test)

# Calculate evaluation metrics
mae_et = mean_absolute_error(y_test, y_pred_et)
mse_et = mean_squared_error(y_test, y_pred_et)
rmse_et = np.sqrt(mse_et)
r2_et = r2_score(y_test, y_pred_et)

print("\nExtra Trees Regressor:")
print("Mean Absolute Error:", round(mae_et, 3))
print("Mean Squared Error:", round(mse_et, 3))
print("Root Mean Squared Error:", round(rmse_et, 3))
print("R-squared:", round(r2_et, 6))


# In[61]:


#BASED ON ABOVE METRICS LINEAR REGRESSION SUITS FOR THE GIVEN DATA SET
#AS GIVEN DATA SET IS SMALL THE CHALLENGES ARE
#OVERFITTING 
#VARIANCE 
#GENERALIZATION


# In[62]:


#RECOMMENDATIONS
# LARGE AMOUNT OF DATA
#CROSS VALIDATION
#ENSEMBLE METHODS
#REGULARIZATION

