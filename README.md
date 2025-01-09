Car Price Prediction Using XGBoost
Overview
This project uses XGBoost, a powerful machine learning algorithm, to predict car prices based on various car attributes. The model analyzes key features such as curb weight, dimensions, engine size, and fuel efficiency to provide accurate price predictions. The project demonstrates the application of feature engineering, hyperparameter tuning, and model evaluation techniques to solve a real-world regression problem.

Objectives
Build a machine learning model to predict car prices based on historical data.
Identify the most important features that influence car prices.
Optimize the model using techniques like GridSearchCV for hyperparameter tuning.
Visualize feature importance and key findings.
Features in the Dataset
The dataset includes the following key features:

Curb Weight (curbweight): The weight of the car without passengers or cargo.
Symboling: A categorical variable representing the car's risk factor or brand attributes.
Wheelbase: Distance between the front and rear axles of the car.
Car Dimensions:
Length (carlength)
Height (carheight)
Width (carwidth)
City MPG (citympg): Fuel efficiency in city driving.
Engine Size (enginesize): Size of the car's engine, influencing performance and price.
Horsepower: Engine power measured in HP.
Compression Ratio: Ratio of the engine's cylinder volume, which affects efficiency.
Tools and Technologies Used
Python for data analysis and model building
XGBoost for building the regression model
Matplotlib and Seaborn for visualizations
Scikit-learn for preprocessing, feature scaling, and hyperparameter tuning
Workflow
Data Preprocessing:
Handling missing values
Encoding categorical variables
Normalizing numerical features
Feature Engineering:
Identifying the most relevant predictors for price
Model Building:
Training the XGBoost model
Optimizing hyperparameters using GridSearchCV
Model Evaluation:
Measuring performance using metrics like:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
Feature Importance Analysis:
Visualizing the top features impacting the model's predictions
Key Results
Highest Impact Features:
Curbweight: The most significant factor influencing car prices.
Symboling: A strong predictor of price related to car risk or brand.
Wheelbase and Car Dimensions: Crucial for understanding car size and luxury levels.
Model Accuracy: Achieved a high R² score (e.g., ~0.9) on the test set, indicating strong predictive performance.
Visualizations
The project includes visualizations such as:

Feature importance bar plots
Scatter plots for actual vs. predicted prices
Correlation heatmaps to show relationships between features
