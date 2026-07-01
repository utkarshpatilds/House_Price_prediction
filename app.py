"""
🏠 House Price Predictor — Professional Streamlit Web App
Predict house prices based on property features | Built by Utkarsh Patil
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom theme via CSS
st.markdown("""
<style>
    /* ── Main bg & text ── */
    .stApp { background-color: #0E1117; }

    /* ── Headers ── */
    h1, h2, h3 { color: #FFFFFF !important; }

    /* ── Metric cards ── */
    div[data-testid="stMetricValue"] { color: #4ECDC4 !important; font-size: 2rem !important; }
    div[data-testid="stMetricLabel"] { color: #A0AEC0 !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F2E 0%, #0E1117 100%);
        border-right: 1px solid #2D3748;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4 0%, #2C9A92 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5FE0D6 0%, #3AA99F 100%);
        box-shadow: 0 4px 20px rgba(78,205,196,0.4);
    }

    /* ── Success / error boxes ── */
    .success-box {
        background: linear-gradient(135deg, rgba(78,205,196,0.15), rgba(78,205,196,0.05));
        border: 1px solid #4ECDC4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,107,107,0.05));
        border: 1px solid #FF6B6B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Feature cards ── */
    .feature-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    /* ── Code / json blocks ── */
    .streamlit-expanderHeader { background: #1A1F2E; border-radius: 8px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1F2E;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #A0AEC0;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #4ECDC4; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #2C9A92 100%);
        color: white;
    }

    /* ── Divider ── */
    hr { border-color: #2D3748; }

    /* ── Footer ── */
    .footer { text-align: center; color: #4A5568; font-size: 0.8rem; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    lr_model = joblib.load("linear_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    with open("model_results.json") as f:
        results = json.load(f)
    return model, lr_model, feature_names, results

model, lr_model, feature_names, results = load_artifacts()

# Load raw data for EDA
@st.cache_data
def load_raw_data():
    return pd.read_csv("Housing.csv")

df = load_raw_data()
df['price_lakhs'] = df['price'] / 100000

FEATURE_GROUPS = {
    "Property Basics": ["area", "bedrooms", "bathrooms", "stories", "parking"],
    "Location & Access": ["mainroad", "guestroom", "prefarea"],
    "Amenities": ["basement", "hotwaterheating", "airconditioning"],
    "Furnishing": ["furnishingstatus"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")
    st.markdown("**Built by:** Utkarsh Patil")
    st.markdown("**GitHub:** [utkarshpatilds](https://github.com/utkarshpatilds)")
    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        [
            "🏠 Home",
            "💰 Price Prediction",
            "📊 EDA Dashboard",
            "📈 Model Performance",
            "🔍 Feature Insights",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div class='footer'>🏗️ Built with Python, Scikit-learn & Streamlit</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("# 🏠 House Price Predictor")
        st.markdown("### Predict Property Values with Machine Learning")
        st.markdown("---")
        st.markdown("""
        Welcome! This interactive web application predicts **house prices** based on
        property features using machine learning — built end-to-end by **Utkarsh Patil**.

        **What you can do here:**
        - 🏠 Enter property details → get an **instant price estimate**
        - 📊 Explore the **Exploratory Data Analysis** dashboard
        - 📈 Compare **Linear Regression vs Random Forest** models
        - 🔍 Understand which **features drive house prices** most
        """)
        st.markdown("---")
        st.markdown("##### Quick Stats")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Total Properties", "545")
        with k2:
            st.metric("Features Used", "12")
        with k3:
            st.metric("Best R² Score", f"{results['LinearRegression']['R2']:.1%}")

        st.markdown("---")
        st.markdown("##### Dataset Features")
        features_html = ""
        for group, feats in FEATURE_GROUPS.items():
            features_html += f"<b>{group}:</b> {', '.join(feats)}  \n"
        st.markdown(features_html)

    with col_right:
        st.markdown("### 🎯 Model Results")
        st.dataframe(
            pd.DataFrame(results).drop('best_model', errors='ignore').drop('feature_names', errors='ignore').T,
            use_container_width=True,
            height=200
        )
        st.markdown("""
        <div class='success-box'>
            🏆 <b>Best Model:</b> Linear Regression<br>
            <span style="font-size:1.5rem; color:#4ECDC4">R² = 65.3%</span><br>
            <small>Explains 65% of price variation</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        <table style="color:#A0AEC0; font-size:0.9rem">
        <tr><td>🤖 ML Models</td><td>Linear Regression, Random Forest</td></tr>
        <tr><td>📊 EDA</td><td>Pandas, Matplotlib, Seaborn</td></tr>
        <tr><td>🌐 Web App</td><td>Streamlit</td></tr>
        <tr><td>🐍 Language</td><td>Python 3</td></tr>
        </table>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰 Price Prediction":
    st.title("💰 House Price Prediction")
    st.markdown("Fill in the property details below to get an **instant price estimate** powered by Linear Regression.")

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### 🏗️ Property Basics")
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.number_input("📐 Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100, help="Total carpet area of the property")
        with c2:
            bedrooms = st.selectbox("🛏️ Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        with c3:
            bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4], index=1)

        c4, c5, c6 = st.columns(3)
        with c4:
            stories = st.selectbox("📶 Stories", [1, 2, 3, 4], index=1)
        with c5:
            parking = st.selectbox("🅿️ Parking Spaces", [0, 1, 2, 3], index=1)
        with c6:
            prefarea = st.radio("📍 Preferred Area?", ["no", "yes"], horizontal=True, index=0)

        st.markdown("### 📍 Location & Access")
        c7, c8 = st.columns(2)
        with c7:
            mainroad = st.radio("🛣️ On Main Road?", ["yes", "no"], horizontal=True, index=0)
        with c8:
            guestroom = st.radio("🛋️ Guestroom?", ["yes", "no"], horizontal=True, index=1)

        st.markdown("### 🔧 Amenities")
        c9, c10, c11 = st.columns(3)
        with c9:
            basement = st.radio("🏚️ Has Basement?", ["yes", "no"], horizontal=True, index=1)
        with c10:
            hotwaterheating = st.radio("🚿 Hot Water Heating?", ["yes", "no"], horizontal=True, index=1)
        with c11:
            airconditioning = st.radio("❄️ Air Conditioning?", ["yes", "no"], horizontal=True, index=0)

        c12, = st.columns(1)
        with c12:
            furnishingstatus = st.select_slider(
                "🛋️ Furnishing Status",
                options=["furnished", "semi-furnished", "unfurnished"],
                value="semi-furnished"
            )

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        # Build input dict matching training columns
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus,
        }

        df_input = pd.DataFrame([input_data])

        # One-hot encode to match training (drop_first=True)
        df_encoded = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_names]

        # Predict
        pred_lr = lr_model.predict(df_encoded)[0]
        pred_rf = model.predict(df_encoded)[0]
        pred_avg = (pred_lr + pred_rf) / 2

        price_lr = pred_lr
        price_rf = pred_rf
        price_avg = pred_avg

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Linear Regression", f"₹ {price_lr:,.0f}")
        with col_p2:
            st.metric("Random Forest", f"₹ {price_rf:,.0f}")
        with col_p3:
            st.metric("Ensemble Average", f"₹ {price_avg:,.0f}")

        st.markdown("---")

        # Price range estimate
        low = min(price_lr, price_rf)
        high = max(price_lr, price_rf)
        st.markdown(f"""
        <div class='success-box'>
            🏷️ <b>Estimated Price Range:</b><br>
            <span style="font-size:2rem; color:#4ECDC4; font-weight:bold">
                ₹ {low:,.0f} — ₹ {high:,.0f}
            </span>
            <br><br>
            📊 Our Linear Regression model predicts this property based on {len(input_data)} features
            including area, amenities, location, and furnishing status.
        </div>
        """, unsafe_allow_html=True)

        # Price per sq ft
        price_per_sqft = pred_lr / area
        area_bucket = "Budget" if area < 4000 else "Mid-range" if area < 7000 else "Premium"
        st.info(f"📐 Price per sq ft: **₹ {price_per_sqft:,.0f}** | Property Class: **{area_bucket}**")


# ─────────────────────────────────────────────────────────────────────────────
# EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Insights from the **Housing dataset** — 545 properties analyzed.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Distribution",
        "🔥 Correlations",
        "🏠 By Feature",
        "📋 Summary Stats"
    ])

    with tab1:
        st.subheader("Price Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1A1F2E')
        for ax in axes:
            ax.set_facecolor('#1A1F2E')
            ax.tick_params(colors='#A0AEC0')
            ax.xaxis.label.set_color('#A0AEC0')
            ax.yaxis.label.set_color('#A0AEC0')
            ax.title.set_color('#FFFFFF')
            ax.spines['bottom'].set_color('#2D3748')
            ax.spines['left'].set_color('#2D3748')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[0].hist(df['price_lakhs'], bins=30, color='#4ECDC4', alpha=0.8, edgecolor='white')
        axes[0].set_title("Price Distribution (in Lakhs)", fontsize=13, pad=10)
        axes[0].set_xlabel("Price (₹ Lakhs)")
        axes[0].set_ylabel("Count")

        axes[1].hist(np.log1p(df['price']), bins=30, color='#FF6B6B', alpha=0.8, edgecolor='white')
        axes[1].set_title("Log-Transformed Price Distribution", fontsize=13, pad=10)
        axes[1].set_xlabel("Log(Price)")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Mean Price", f"₹ {df['price_lakhs'].mean():.1f} L")
        with col_b:
            st.metric("Median Price", f"₹ {df['price_lakhs'].median():.1f} L")
        with col_a:
            st.metric("Most Expensive", f"₹ {df['price_lakhs'].max():.1f} L")
        with col_b:
            st.metric("Most Affordable", f"₹ {df['price_lakhs'].min():.1f} L")

    with tab2:
        st.subheader("Correlation Heatmap")
        df_num = df.select_dtypes(include=[np.number])
        corr = df_num.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#1A1F2E')
        ax.set_facecolor('#1A1F2E')
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax,
            annot_kws={"color": "white", "size": 9},
            cbar_kws={'label': 'Correlation', 'label_color': 'white'},
            linewidths=0.5, linecolor='#2D3748'
        )
        ax.tick_params(colors='#A0AEC0', labelsize=9)
        ax.set_title("Feature Correlation Heatmap", color='white', pad=15, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Top Features Correlated with Price")
        price_corr = corr['price'].drop('price').sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor('#1A1F2E')
        ax2.set_facecolor('#1A1F2E')
        colors = ['#4ECDC4' if v > 0 else '#FF6B6B' for v in price_corr.values]
        price_corr.plot(kind='barh', ax=ax2, color=colors, edgecolor='white')
        ax2.set_title("Correlation with Price", color='white', pad=10)
        ax2.set_xlabel("Correlation Coefficient", color='#A0AEC0')
        ax2.tick_params(colors='#A0AEC0')
        ax2.spines['bottom'].set_color('#2D3748')
        ax2.spines['left'].set_color('#2D3748')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.axvline(0, color='#2D3748', linewidth=1)
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.subheader("Price Breakdown by Feature")

        feature_to_plot = st.selectbox(
            "Select a feature to analyze:",
            ["bedrooms", "bathrooms", "stories", "parking", "mainroad",
             "guestroom", "basement", "hotwaterheating", "airconditioning",
             "prefarea", "furnishingstatus"]
        )

        if df[feature_to_plot].dtype == 'object':
            group_data = df.groupby(feature_to_plot)['price_lakhs'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1A1F2E')
            ax.set_facecolor('#1A1F2E')
            group_data.plot(kind='bar', ax=ax, color='#4ECDC4', edgecolor='white')
            ax.set_title(f"Average Price by {feature_to_plot.title()}", color='white', pad=10)
            ax.set_ylabel("Avg Price (₹ Lakhs)", color='#A0AEC0')
            ax.tick_params(colors='#A0AEC0', labelrotation=0)
            ax.spines['bottom'].set_color('#2D3748')
            ax.spines['left'].set_color('#2D3748')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(ax.patches, group_data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'₹{val:.1f}L', ha='center', color='#A0AEC0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1A1F2E')
            ax.set_facecolor('#1A1F2E')
            ax.scatter(df[feature_to_plot], df['price_lakhs'], alpha=0.5, c='#4ECDC4', s=40)
            ax.set_title(f"Price vs {feature_to_plot.title()}", color='white', pad=10)
            ax.set_xlabel(feature_to_plot.title(), color='#A0AEC0')
            ax.set_ylabel("Price (₹ Lakhs)", color='#A0AEC0')
            ax.tick_params(colors='#A0AEC0')
            ax.spines['bottom'].set_color('#2D3748')
            ax.spines['left'].set_color('#2D3748')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Area vs Price scatter
        st.subheader("Area vs Price")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        fig3.patch.set_facecolor('#1A1F2E')
        ax3.set_facecolor('#1A1F2E')
        ax3.scatter(df['area'], df['price_lakhs'], alpha=0.5, c='#4ECDC4', s=40)
        z = np.polyfit(df['area'], df['price_lakhs'], 1)
        p = np.poly1d(z)
        ax3.plot(df['area'].sort_values(), p(df['area'].sort_values()), color='#FF6B6B', linewidth=2, label='Trend line')
        ax3.set_title("Area vs Price — Strongest Correlation", color='white', pad=10)
        ax3.set_xlabel("Area (sq ft)", color='#A0AEC0')
        ax3.set_ylabel("Price (₹ Lakhs)", color='#A0AEC0')
        ax3.tick_params(colors='#A0AEC0')
        ax3.legend(facecolor='#1A1F2E', labelcolor='#A0AEC0')
        ax3.spines['bottom'].set_color('#2D3748')
        ax3.spines['left'].set_color('#2D3748')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

    with tab4:
        st.subheader("Statistical Summary")
        st.dataframe(
            df.describe().style.background_gradient(cmap='coolwarm', vmin=0, vmax=df['price'].max()),
            use_container_width=True
        )
        st.markdown("---")
        st.subheader("Categorical Feature Counts")
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            counts = df[col].value_counts()
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#1A1F2E')
            ax.set_facecolor('#1A1F2E')
            counts.plot(kind='bar', ax=ax, color='#4ECDC4', edgecolor='white')
            ax.set_title(f"Distribution: {col.title()}", color='white')
            ax.tick_params(colors='#A0AEC0', labelrotation=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#2D3748')
            ax.spines['left'].set_color('#2D3748')
            plt.tight_layout()
            st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Comparison")

    st.markdown("""
    Two regression models were trained on **80%** of the data and evaluated on **20%** held-out test data.
    Both models were compared using industry-standard regression metrics.
    """)

    # Metrics table
    metrics_df = pd.DataFrame(results).drop('best_model', errors='ignore').drop('feature_names', errors='ignore').T
    metrics_df.index.name = 'Model'
    st.dataframe(
        metrics_df.style
            .background_gradient(subset=['R2'], cmap='Greens')
            .background_gradient(subset=['MAE', 'RMSE'], cmap='Reds'),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 📊 Metric Explanation")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("""
        **MAE — Mean Absolute Error**
        Average absolute difference between predicted and actual prices.
        Lower = better. Unit: ₹
        """)
    with col_e2:
        st.markdown("""
        **RMSE — Root Mean Squared Error**
        Penalizes large errors more heavily. Lower = better. Unit: ₹
        """)
    with col_e3:
        st.markdown("""
        **R² — Coefficient of Determination**
        Explains how much of price variation the model captures. Higher = better (max 1.0).
        """)

    st.markdown("---")
    st.subheader("Model Comparison Chart")

    # Bar chart comparison
    models_list = list(results.keys())
    mae_vals = [results[m]['MAE'] for m in models_list]
    rmse_vals = [results[m]['RMSE'] for m in models_list]
    r2_vals = [results[m]['R2'] for m in models_list]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1A1F2E')
    for ax in axes:
        ax.set_facecolor('#1A1F2E')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2D3748')
        ax.spines['left'].set_color('#2D3748')
        ax.tick_params(colors='#A0AEC0')

    colors = ['#4ECDC4', '#FF6B6B']
    labels = ['Linear Regression', 'Random Forest']

    axes[0].bar(labels, mae_vals, color=colors, edgecolor='white')
    axes[0].set_title("MAE (Lower is Better)", color='white', pad=10)
    axes[0].set_ylabel("₹ Error", color='#A0AEC0')
    for bar, val in zip(axes[0].patches, mae_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                     f'₹{val/1e6:.2f}M', ha='center', color='#A0AEC0', fontsize=9)

    axes[1].bar(labels, rmse_vals, color=colors, edgecolor='white')
    axes[1].set_title("RMSE (Lower is Better)", color='white', pad=10)
    axes[1].set_ylabel("₹ Error", color='#A0AEC0')
    for bar, val in zip(axes[1].patches, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                     f'₹{val/1e6:.2f}M', ha='center', color='#A0AEC0', fontsize=9)

    axes[2].bar(labels, r2_vals, color=colors, edgecolor='white')
    axes[2].set_title("R² Score (Higher is Better)", color='white', pad=10)
    axes[2].set_ylabel("R² Score", color='#A0AEC0')
    axes[2].set_ylim(0, 1)
    for bar, val in zip(axes[2].patches, r2_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', color='#A0AEC0', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    <div class='success-box'>
        ✅ <b>Winner: Linear Regression</b> with R² = 65.3%<br>
        <small>Linear Regression outperforms Random Forest on this dataset, explaining 65% of price variation.
        The relationship between features and price is largely linear in this data.</small>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Feature Insights":
    st.title("🔍 Feature Importance Analysis")
    st.markdown("""
    Understanding which property features have the **biggest impact on price** — from the Random Forest model.
    """)

    # Feature importance from RF
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Correlation with price
    df_full = df.copy()
    df_full_encoded = pd.get_dummies(df_full, drop_first=True)
    price_corr = df_full_encoded.corr()['price'].drop('price').sort_values(ascending=False)

    tab_f1, tab_f2 = st.tabs(["🌲 Random Forest Importance", "📊 Correlation with Price"])

    with tab_f1:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#1A1F2E')
        ax.set_facecolor('#1A1F2E')
        ax.barh(rf_importance['Feature'], rf_importance['Importance'],
                color='#4ECDC4', edgecolor='white')
        ax.set_title("Feature Importance — Random Forest", color='white', pad=15, fontsize=14)
        ax.set_xlabel("Importance Score", color='#A0AEC0')
        ax.tick_params(colors='#A0AEC0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2D3748')
        ax.spines['left'].set_color('#2D3748')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Top Price Drivers")
        top_features = rf_importance.head(4)
        for i, row in top_features.iterrows():
            pct = row['Importance'] * 100
            st.markdown(f"""
            <div style="background:#1A1F2E; border:1px solid #2D3748; border-radius:8px; padding:1rem; margin:0.5rem 0">
                <b style="color:#4ECDC4">{row['Feature'].title()}</b> — {pct:.1f}% of model decision
            </div>
            """, unsafe_allow_html=True)

    with tab_f2:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#1A1F2E')
        ax.set_facecolor('#1A1F2E')
        colors = ['#4ECDC4' if v > 0 else '#FF6B6B' for v in price_corr.values]
        price_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        ax.set_title("Feature Correlation with Price", color='white', pad=15, fontsize=14)
        ax.set_xlabel("Pearson Correlation", color='#A0AEC0')
        ax.tick_params(colors='#A0AEC0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2D3748')
        ax.spines['left'].set_color('#2D3748')
        ax.axvline(0, color='#2D3748', linewidth=1)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("💡 Key Business Insights")
    insights = [
        ("📐", "Area is the #1 price driver", "Larger properties command exponentially higher prices — area alone accounts for 46.8% of Random Forest decisions."),
        ("🛁", "Bathrooms matter more than bedrooms", "Homes with more bathrooms are priced significantly higher, often more so than bedroom count."),
        ("❄️", "Air conditioning adds premium value", "AC-equipped homes show notable price uplift — a clear amenity premium in the market."),
        ("🅿️", "Parking is a strong differentiator", "Parking availability is among the top 5 price drivers, reflecting urban buyer priorities."),
        ("📶", "Stories increase value linearly", "Multi-story homes command higher prices — vertical expansion adds clear value."),
        ("🛋️", "Furnishing status impacts price", "Furnished properties trade at a premium vs semi-furnished and unfurnished listings."),
    ]
    for icon, title, desc in insights:
        st.markdown(f"**{icon} {title}**  \n{desc}\n")
