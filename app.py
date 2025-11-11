import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import time
from datetime import datetime

# ===== SIMPLE AUTHENTICATION =====
USER_CREDENTIALS = {
    "admin": "password123",
    "analyst": "password123", 
    "viewer": "password123"
}

def check_login(username, password):
    return USER_CREDENTIALS.get(username) == password

# ===== PAGE SETUP =====
st.set_page_config(page_title="Polio Dashboard", page_icon="💉", layout="wide")

# ===== CUSTOM STYLING =====
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOGIN PAGE =====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown('<div class="main-title">🔐 Polio Vaccine Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.info("**Demo Logins:** admin / analyst / viewer | Password: password123")
    st.stop()

# ===== MAIN DASHBOARD =====
def load_data():
    return pd.read_csv('new cleaned file (1).csv', index_col=0)

df = load_data()

# Sidebar
st.sidebar.title(f"Welcome, {st.session_state.username}!")
analysis_type = st.sidebar.selectbox("Choose Analysis", [
    "Dashboard", "Trends", "Countries", "Statistics"
])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Main Content
st.markdown('<div class="main-title">💉 Polio Vaccine Coverage</div>', unsafe_allow_html=True)

# Key Metrics
if 'Share polio vaccine (POL3)' in df.columns:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Coverage", f"{df['Share polio vaccine (POL3)'].mean():.1f}%")
    with col2:
        st.metric("Countries", df['Entity'].nunique())
    with col3:
        st.metric("Years", f"{df['Year'].min()}-{df['Year'].max()}")
    with col4:
        st.metric("Max Coverage", f"{df['Share polio vaccine (POL3)'].max():.1f}%")

# Analysis Sections
if analysis_type == "Dashboard":
    st.subheader("📊 Quick Overview")
    st.dataframe(df.head(10))
    
elif analysis_type == "Trends":
    st.subheader("📈 Coverage Trends")
    if all(col in df.columns for col in ['Entity', 'Year', 'Share polio vaccine (POL3)']):
        countries = st.multiselect("Select Countries", df['Entity'].unique(), default=df['Entity'].unique()[:3])
        if countries:
            plot_df = df[df['Entity'].isin(countries)]
            fig = px.line(plot_df, x='Year', y='Share polio vaccine (POL3)', color='Entity')
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Countries":
    st.subheader("🌍 Country Comparison")
    if 'Entity' in df.columns:
        top_countries = df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(10)
        fig = px.bar(x=top_countries.values, y=top_countries.index, orientation='h')
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Statistics":
    st.subheader("📐 Statistical Analysis")
    if 'Share polio vaccine (POL3)' in df.columns:
        st.dataframe(df['Share polio vaccine (POL3)'].describe())

# Footer
st.markdown("---")
st.markdown("Polio Vaccine Dashboard • Secure Access")
