# 🏠 House Price Prediction | Machine Learning Web App



A modern **Machine Learning web application** built with **Python, Streamlit, and Scikit-learn** that predicts house prices based on property features. The application also provides an interactive dashboard for exploratory data analysis (EDA), model comparison, and feature importance visualization.

---

## 🚀 Live Demo

🔗 **Live Application:** https://house-price-prediction-ecp6.onrender.com/
---

# ✨ Features

- 🏠 Predict house prices using Machine Learning
- 📊 Interactive Exploratory Data Analysis (EDA)
- 📈 Model Performance Comparison
- 📉 Correlation Heatmap
- 📋 Summary Statistics
- 🎯 Feature Importance Analysis
- 🌙 Modern Dark-Themed UI
- 📱 Responsive Streamlit Dashboard
- ⚡ Real-time Predictions

---

# 🧠 Machine Learning Pipeline

### Data Collection

- Housing Dataset
- 545 Property Records

### Data Preprocessing

- Missing Value Check
- Feature Encoding
- Feature Scaling
- Train-Test Split (80:20)

### Models Used

- Linear Regression
- Random Forest Regressor

### Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# 📊 Exploratory Data Analysis

The application provides multiple interactive visualizations:

- Price Distribution
- Feature Distribution
- Correlation Heatmap
- Summary Statistics
- Feature-wise Analysis

---

# 📈 Model Performance

| Model | R² Score | MAE | RMSE |
|------|---------:|---------:|---------:|
| Linear Regression | **65.3%** | ₹97,004 | ₹132,451 |
| Random Forest | 61.2% | ₹102,546 | ₹140,565 |

🏆 **Best Performing Model:** Linear Regression

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Web Framework

- Streamlit

### Model Serialization

- Joblib
- Pickle

---

# 📂 Project Structure

```
House_Price_prediction/
│
├── Housing.csv
├── app.py
├── analysis.ipynb
├── best_model.pkl
├── linear_model.pkl
├── feature_names.pkl
├── model_results.json
├── requirements.txt
├── README.md
└── render.yaml
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/House_Price_prediction.git
```


# 📌 Input Features

The prediction model uses the following features:

### Property Information

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking

### Location

- Main Road Access
- Guest Room
- Preferred Area

### Amenities

- Basement
- Hot Water Heating
- Air Conditioning

### Furnishing

- Furnishing Status

---

# 📊 Model Workflow

```
Housing Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Saved
        │
        ▼
Streamlit Web Application
```

---

# 🎯 Future Improvements

- Deep Learning Models
- XGBoost Regressor
- CatBoost Regressor
- LightGBM
- Hyperparameter Tuning
- Model Explainability (SHAP)
- User Authentication
- Database Integration
- Prediction History
- PDF Report Generation

---

# 👨‍💻 Author

**Utkarsh Patil**

- GitHub: https://github.com/utkarshpatilds
- LinkedIn: * https://www.linkedin.com/in/utkarshpatil-ds/ *

---


