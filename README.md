# House_Price_prediction
Project Overview

This project aims to predict house prices based on various property features such as area, number of bedrooms, bathrooms, stories, parking availability, furnishing status, and other amenities. The project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), model building, evaluation, and visualization.
The goal is to identify the key factors that influence house prices and build predictive models that can estimate property values accurately.

Dataset Information

The dataset contains 545 residential properties with multiple features describing each house.

Target Variable :

Price – House selling price

Features :
Area
Bedrooms
Bathrooms
Stories
Main Road Access
Guest Room
Basement
Hot Water Heating
Air Conditioning
Parking
Preferred Area
Furnishing Status
And other property-related attributes


Project Workflow

1. Data Collection
Loaded the housing dataset using Pandas.

2. Data Exploration
Examined dataset shape and structure.
Checked column names and data types.
Generated descriptive statistics.
Verified data quality.

3. Data Cleaning
Checked for missing values.
Checked for duplicate records.
Confirmed the dataset was clean and ready for modeling.

4. Data Preprocessing
Converted categorical variables into numerical format using One-Hot Encoding (pd.get_dummies()).
Separated features and target variable.

5. Train-Test Split
Split the dataset into:
80% Training Data
20% Testing Data
Used a fixed random state for reproducibility.

6. Model Building

Linear Regression
A baseline regression model was trained to understand linear relationships between features and house prices.

Random Forest Regressor
An ensemble learning model was trained to capture complex and non-linear relationships within the dataset.

7. Model Evaluation
The following metrics were used:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score (Coefficient of Determination)

8. Data Visualization
The project includes:
House Price Distribution
Correlation Heatmap
Actual vs Predicted Prices
Feature Importance Analysis

Technologies Used :
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Jupyter Notebook


Project Structure
House-Price-Prediction/
│
├── Housing.csv
├── House_Price_Prediction.ipynb
├── README.md
│
└── images/
    ├── price_distribution.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── feature_importance.png


Key Findings
The dataset contained 545 records with no missing values and no duplicate entries.
Property area showed the strongest relationship with house prices.
Other important features included:
Bathrooms
Air Conditioning
Number of Stories
Parking Availability
Preferred Area

Houses with larger areas and additional amenities generally had higher market values.
Feature importance analysis confirmed that property size and amenities significantly influence pricing.

Results
Two machine learning models were trained and evaluated:

| Model                   | Purpose                   |
| ----------------------- | ------------------------- |
| Linear Regression       | Baseline predictive model |
| Random Forest Regressor | Non-linear ensemble model |

The model performance was evaluated using MAE, RMSE, and R² Score to determine prediction accuracy.


Business Impact
A house price prediction system can help:
Real estate companies estimate property values.
Buyers make informed purchasing decisions.
Sellers set competitive prices.
Investors identify high-value properties.
Banks evaluate collateral value during loan approvals.
Future Improvements
Hyperparameter tuning using GridSearchCV.
Cross-validation for robust evaluation.
Outlier detection and treatment.
Feature engineering.

Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline for house price prediction. After cleaning and preprocessing the data, multiple regression models were trained and evaluated. The analysis revealed that factors such as property area, bathrooms, parking facilities, and air conditioning play a major role in determining house prices. The project provides practical insights into real estate valuation while showcasing essential data analytics and machine learning skills.


Author :
Utkarsh Patil
B.Tech CSE (Data Science)
