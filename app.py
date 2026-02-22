import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Configuration
st.set_page_config(
    page_title="AI Business Automation Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Loading Functions
@st.cache_data
def load_sales_data():
    path = os.path.join(BASE_DIR, 'sales_data.csv')
    df = pd.read_csv(path, encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    if df['Order Date'].isnull().sum() > len(df) * 0.5:
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=False)
    
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['MonthName'] = df['Order Date'].dt.month_name()
    return df

@st.cache_data
def load_churn_data():
    path = os.path.join(BASE_DIR, 'churn_data.csv')
    df = pd.read_csv(path)
    # Preprocess TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

# Helper to save/load models or run quick training
def train_quick_sales_model(df):
    # Simplified training for demo purposes
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
    daily_sales['DayOfYear'] = daily_sales['Order Date'].dt.dayofyear
    daily_sales['Year'] = daily_sales['Order Date'].dt.year
    
    X = daily_sales[['DayOfYear', 'Year']]
    y = daily_sales['Sales']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, daily_sales

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("AI Business Suite")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Go to", ["Executive Overview", "Sales Forecasting", "Customer Churn Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Developed by Iffat Nazir")

# --- Page 1: Executive Overview ---
if page == "Executive Overview":
    st.title("🚀 AI Business Automation Suite")
    st.markdown("""
    This interactive dashboard empowers business leaders to make data-driven decisions using state-of-the-art AI.
    Select a module from the sidebar to begin your analysis.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sales Forecasting")
        st.write("Predict revenue trends and identify growth opportunities across regions and categories.")
        if st.button("Explore Sales"):
            st.session_state.page = "Sales Forecasting" # Note: Simple state management might be needed
            st.rerun()

    with col2:
        st.subheader("👥 Customer Churn")
        st.write("Identify at-risk customers and understand the key drivers of churn to improve retention.")
        if st.button("Explore Churn"):
            st.session_state.page = "Customer Churn Prediction"
            st.rerun()

    st.markdown("---")
    st.image(os.path.join(BASE_DIR, "charts/dashboard/01_sales_executive_dashboard.png"), caption="Sample Executive Report", use_container_width=True)

# --- Page 2: Sales Forecasting ---
elif page == "Sales Forecasting":
    st.title("📊 Sales Forecasting & Analytics")
    sales_df = load_sales_data()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_rev = sales_df['Sales'].sum()
    avg_order = sales_df['Sales'].mean()
    unique_cust = sales_df['Customer ID'].nunique()
    total_orders = len(sales_df)

    col1.metric("Total Revenue", f"${total_rev:,.0f}")
    col2.metric("Avg Order Value", f"${avg_order:,.2f}")
    col3.metric("Total Customers", f"{unique_cust:,}")
    col4.metric("Total Orders", f"{total_orders:,}")
    
    st.markdown("---")
    
    # Interactivity
    st.subheader("🔍 Interactive Analysis")
    category_filter = st.multiselect("Select Category", options=sales_df['Category'].unique(), default=sales_df['Category'].unique())
    filtered_df = sales_df[sales_df['Category'].isin(category_filter)]
    
    # Visualizations
    tab1, tab2 = st.tabs(["Trend Analysis", "Forecasting"])
    
    with tab1:
        st.markdown("### Monthly Sales Trend")
        monthly_sales = filtered_df.groupby(['Year', 'Month', 'MonthName'])['Sales'].sum().reset_index()
        fig_trend = px.line(monthly_sales, x='MonthName', y='Sales', color='Year', 
                            title="Revenue Trend by Year", markers=True,
                            category_orders={"MonthName": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]})
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### Sales by Category")
            fig_pie = px.pie(filtered_df, values='Sales', names='Category', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_c2:
            st.markdown("### Regional Performance")
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            fig_bar = px.bar(region_sales, x='Region', y='Sales', color='Region', text_auto='.2s')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("### AI-Driven Sales Forecast")
        model, daily_sales = train_quick_sales_model(filtered_df)
        
        # Simple prediction for next 30 days
        last_date = daily_sales['Order Date'].max()
        future_dates = pd.date_range(start=last_date, periods=30)
        future_df = pd.DataFrame({'Order Date': future_dates})
        future_df['DayOfYear'] = future_df['Order Date'].dt.dayofyear
        future_df['Year'] = future_df['Order Date'].dt.year
        
        preds = model.predict(future_df[['DayOfYear', 'Year']])
        future_df['Forecast'] = preds
        
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=daily_sales['Order Date'][-90:], y=daily_sales['Sales'][-90:], name="Actual"))
        fig_fc.add_trace(go.Scatter(x=future_df['Order Date'], y=future_df['Forecast'], name="AI Forecast", line=dict(dash='dash', color='red')))
        fig_fc.update_layout(title="30-Day Sales Forecast (Random Forest)", xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig_fc, use_container_width=True)
        st.success("AI Model successfully forecasted future trends based on historical seasonality.")

# --- Page 3: Customer Churn Prediction ---
elif page == "Customer Churn Prediction":
    st.title("👥 Customer Churn Analytics")
    churn_df = load_churn_data()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_cust = len(churn_df)
    churn_rate = (churn_df['Churn'] == 'Yes').mean() * 100
    avg_monthly = churn_df['MonthlyCharges'].mean()
    high_revenue_churn = churn_df[(churn_df['Churn'] == 'Yes') & (churn_df['MonthlyCharges'] > 80)].shape[0]

    col1.metric("Total Customers", f"{total_cust:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Monthly Bill", f"${avg_monthly:.2f}")
    col4.metric("High-Risk High-Rev", f"{high_revenue_churn}")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2 = st.tabs(["Churn Drivers", "Risk Calculator"])
    
    with tab1:
        st.markdown("### Key Churn Factors")
        col_cv1, col_cv2 = st.columns(2)
        
        with col_cv1:
            st.markdown("#### Contract Type vs Churn")
            fig_contract = px.histogram(churn_df, x='Contract', color='Churn', barmode='group', 
                                        color_discrete_map={'No': '#4ECDC4', 'Yes': '#FF6B6B'})
            st.plotly_chart(fig_contract, use_container_width=True)
        
        with col_cv2:
            st.markdown("#### Monthly Charges Distribution")
            fig_hist = px.histogram(churn_df, x='MonthlyCharges', color='Churn', marginal="box",
                                    color_discrete_map={'No': '#4ECDC4', 'Yes': '#FF6B6B'})
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.markdown("#### Tenure vs Monthly Charges (Risk Heatmap)")
        fig_scatter = px.scatter(churn_df, x='tenure', y='MonthlyCharges', color='Churn',
                                 opacity=0.5, color_discrete_map={'No': '#4ECDC4', 'Yes': '#FF6B6B'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.markdown("### Interactive Churn Risk Calculator")
        st.write("Adjust the parameters to see if a customer is likely to leave.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly Charges ($)", 18, 120, 70)
        with c2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with c3:
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            
        if st.button("Calculate Risk Score"):
            # Dummy logic for the demo (in real app, we'd load the XGBoost model)
            risk_score = 0.0
            if contract == "Month-to-month": risk_score += 0.4
            if internet == "Fiber optic": risk_score += 0.2
            if tenure < 6: risk_score += 0.3
            if monthly > 90: risk_score += 0.1
            
            risk_score = min(risk_score, 0.95)
            
            st.markdown("---")
            if risk_score > 0.5:
                st.error(f"⚠️ High Risk Level: {risk_score:.1%}")
                st.write("Recommendation: Offer a long-term contract discount or high-priority support.")
            else:
                st.success(f"✅ Low Risk Level: {risk_score:.1%}")
                st.write("Recommendation: Maintain current service quality.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 AI Agriculture Suite (Project 7)")
