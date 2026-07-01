# 🏠 House Price Predictor — Live Demo

> Predict house prices based on property features using Machine Learning | Built by **Utkarsh Patil**

**Live Demo:** https://house-price-predictor.onrender.com

---

## 🎯 What This Project Does

- **Predict house prices** — enter property details and get an instant price estimate
- **Interactive EDA dashboard** — explore price distributions, correlations, and breakdowns
- **Model comparison** — Linear Regression vs Random Forest with full metrics
- **Feature insights** — understand which factors drive house prices most

---

## 📁 Project Structure

```
house_price_project/
├── app.py              ← Streamlit web app (main entry point)
├── train_model.py      ← ML training pipeline
├── best_model.pkl      ← Trained Random Forest model
├── linear_model.pkl    ← Trained Linear Regression model
├── feature_names.pkl   ← Feature column names (44 after encoding)
├── model_results.json  ← Model evaluation metrics
├── Housing.csv         ← Dataset (545 properties)
├── requirements.txt    ← Python dependencies
├── render.yaml         ← Render deployment config
└── README.md
```

---

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** 🏆 | ₹9,70,043 | ₹13,24,507 | **0.653** |
| Random Forest | ₹10,21,546 | ₹14,00,566 | 0.610 |

---

## 🚀 Deploy to Render (Free) — 5 Minutes

### Step 1 — Update Your GitHub Repo

1. Go to: https://github.com/utkarshpatilds/House_Price_prediction
2. Click **"Add file"** → **"Upload files"**
3. Drag and drop ALL files from the `house_price_project` folder:
   - `app.py`
   - `best_model.pkl`
   - `linear_model.pkl`
   - `feature_names.pkl`
   - `model_results.json`
   - `Housing.csv`
   - `requirements.txt`
   - `render.yaml`
4. Commit directly to `main` branch

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) → Sign up with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo (`House_Price_prediction`)
4. Fill in settings:

| Field | Value |
|-------|-------|
| **Name** | `house-price-predictor` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Plan** | `Free` |

5. Click **"Create Web Service"** → Wait 2-3 minutes → **Live!** 🎉

---

## 📝 Resume Write-up

```
House Price Prediction | Live Demo: https://house-price-predictor.onrender.com
GitHub: https://github.com/utkarshpatilds/House_Price_prediction

• Built end-to-end ML pipeline to predict house prices (545 properties, 12 features)
• Compared Linear Regression vs Random Forest — Linear Regression deployed (R² = 0.653)
• Built interactive Streamlit web app with EDA dashboard, model comparison, and live predictions
• Key price drivers: Area (46.8%), Bathrooms (15.1%), Air Conditioning, Parking, Stories
• Tech Stack: Python, Scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn
```

---

## 👤 Author

**Utkarsh Patil** — B.Tech CSE (Data Science)
- GitHub: [utkarshpatilds](https://github.com/utkarshpatilds)
- Project: [House_Price_prediction](https://github.com/utkarshpatilds/House_Price_prediction)
