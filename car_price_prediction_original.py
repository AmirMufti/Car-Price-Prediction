# # Car Price Predication

# importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

# loading the dataset
df = pd.read_csv(r'C:\Users\Mr Amir Mufti\Downloads\CarPrice_Assignment.csv')

# viewing the dataframe
pd.set_option('display.max_columns', None)
df.head()

# checking the shape
df.shape

# understanding data types
df.info()

# indentifying null values
df.isnull().sum()

df.describe()

# # Exploratory Data Analysis

plt.figure(figsize=(8, 4))
sns.histplot(df['price'], kde=True, color="skyblue")
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='price', y='enginesize')
plt.title('Prices of cars based on engine size')
plt.xlabel('Price of the car')
plt.ylabel('Engine size')
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='price', y='horsepower')
plt.title('Prices based on horsepower')
plt.xlabel('Prices')
plt.ylabel('Horsepower')
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='price', y='citympg')
plt.title('Prices based on horsepower')
plt.xlabel('Prices')
plt.ylabel('Miles Per Gallon')
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='horsepower', y='citympg')
plt.title('Prices based on horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon')
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='enginesize', y='citympg')
plt.title('Prices based on horsepower')
plt.xlabel('Engine Size')
plt.ylabel('Miles Per Gallon')
plt.show()

# ##### Price has a positive linear relationship with engine size and horsepower, while negative relationship with miles per gallon.

# # Data Wrangling

# checking the mean price
mean_price = df['price'].mean()

# lets identify outliers that are above 3 standard deviation of the mean value
df['price3SD'] = df['price'] + 3*mean_price

df.head()

df[df['price'] > df['price3SD']].head()

# ##### No outlier found in 3 Standard Deviation of the mean value

# # Feature Engineering

# value counts by group
df.groupby('CarName')['CarName'].agg('count').sort_values(ascending=False)

df[['CarBrand', 'CarModel']] = df['CarName'].str.split(" ", n=1, expand=True)

# dropping the uncessary columns
df = df.drop(['car_ID', 'CarName', 'CarModel'], axis= 1)

for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print()

df1 = df.copy()

# Ordinal Encoding for 'symboling'
df['symboling'] = df['symboling'].astype(int)

# Binary Encoding for 'fueltype', 'aspiration', and 'enginelocation'
binary_mappings = {
    'fueltype': {'gas': 0, 'diesel': 1},
    'aspiration': {'std': 0, 'turbo': 1},
    'enginelocation': {'front': 0, 'rear': 1}
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

# One-Hot Encoding for categorical columns
one_hot_cols = ['doornumber', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem', 'CarBrand']
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)

# Normalization/Scaling for numerical columns
scaler = MinMaxScaler()
numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio',
                'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.head()

df.shape

# # Building Prediction Model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Droping the extra columns
df = df.loc[:, ~df.columns.str.startswith('CarBrand_')]
df = df.loc[:, ~df.columns.str.startswith('fuelsystem_')]
df = df.loc[:, ~df.columns.str.startswith('enginetype_')]

X = df.drop(['price', 'price3SD'], axis=1)
y= df1.price

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### Linear Regression

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# #### Other Models Using GridSearchCV

model_params = {
    'LinearRegression': {
        'model' : LinearRegression(),
        'params': {}
    },
    'XGB':{
    'model': xgb.XGBRegressor(),
    'params':{
            'booster': ['gbtree', 'gblinear'],   
            'eta': [0.1, 0.2],    
            'n_estimators' : [50,100],
            'max_depth' : [2,4],
            'learning_rate' : [0.1, 0.05],
            'min_child_weight' : [2,5],
            'lambda': [1.0, 2.0]
            }
        },
        
    'RandomForestRegressor':{
        'model': RandomForestRegressor(),
        'params':{
            'criterion' : ['absolute_error', 'squared_error'],
            'max_depth' : [4, 6],
            'min_samples_leaf': [2, 5]
        }
            },
    'Decision_Tree':{
        'model': DecisionTreeRegressor(),
        'params':{
            'criterion' : ['absolute_error', 'squared_error'],
            'max_depth' : [4, 6],
            'min_samples_leaf': [2, 5]
            }                  
        }
    }

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_,
        })

df1 = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df1

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type="weight", max_num_features=10)  # Show top 10 features
plt.title("Feature Importance (XGBoost)")
plt.show()

# 
# The plot you shared presents the feature importance from an XGBoost model used to predict car prices. It shows how much each feature (predictor variable) contributed to the model's prediction based on their F-scores (frequency and weight). Here's the breakdown of the most important features:
# 1.	Curweight (curbweight):
# o	This feature has the highest importance score (738.0), meaning it played the most significant role in predicting the car price. Curbweight refers to the weight of the car without passengers or cargo, and it is likely to have a strong relationship with the car's overall price.
# 2.	Symboling:
# o	With an importance score of 219.0, symboling likely represents a categorical code related to the car's risk or brand attributes. Higher values might correlate with cars that are higher-priced or considered higher in value.
# 3.	Wheelbase:
# o	Wheelbase, with an importance score of 190.0, is another crucial feature. It refers to the distance between the front and rear axles. Larger wheelbases are often associated with more expensive, larger vehicles.
# 4.	Car Length (carlength):
# o	Car length (importance score of 142.0) indicates how long the car is. Longer cars tend to be higher-priced, especially for luxury or full-size models.
# 5.	Car Height (carheight):
# o	Car height (score 125.0) is less important than car length or wheelbase but still contributes to the model’s prediction. Taller cars, like SUVs, may have a different price range compared to sedans.
# 6.	Car Width (carwidth):
# o	Car width (113.0) is important, reflecting how broad the car is. Wider cars often indicate larger or more luxurious models, which may be priced higher.
# 7.	City MPG (citympg):
# o	With an importance score of 98.0, city MPG (miles per gallon) is related to fuel efficiency. Cars with better fuel efficiency, especially in city driving, could influence price decisions.
# 8.	Engine Size (enginesize):
# o	Engine size (77.0) influences the power and performance of the car, which affects the price. Larger engines are often found in more expensive vehicles.
# 9.	Horsepower:
# o	Horsepower (70.0) is a measure of the engine's power, and this can influence the car's price, particularly in sports or luxury cars.
# 10.	Compression Ratio:
# •	Compression ratio (70.0) plays a smaller role in determining car prices but might still impact the engine's efficiency and performance, particularly for higher-end or performance cars.
# Conclusion:
# The feature importance plot shows that the weight, dimensions, and performance-related features (curbweight, symboling, wheelbase, carlength, carheight, carwidth) are key drivers in predicting car prices. Performance-related features like engine size, horsepower, and fuel efficiency (citympg) also contribute but are less influential than the physical characteristics of the car.
# 
# 


