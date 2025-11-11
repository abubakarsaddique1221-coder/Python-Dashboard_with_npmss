import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io

# --- 1. SET UP THE PAGE WITH MODERN THEME ---
st.set_page_config(
    page_title="Polio Vaccine Dashboard",
    page_icon="💉",
    layout="wide"
)

# --- 2. MODERN CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stSelectbox, .stRadio {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('new cleaned file (1).csv', index_col=0)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 4. SIDEBAR CONFIGURATION ---
st.sidebar.markdown('<div style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
st.sidebar.markdown('### 💉 Polio Vaccine Dashboard')
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Load data
df = load_data()

if df is None:
    st.error("Could not load data. Please check if the file exists.")
    st.stop()

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "📊 Analysis Type",
    [
        "Data Overview",
        "Vaccine Trends",
        "Country Comparison",
        "Statistical Analysis"
    ]
)

# Quick filters
st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 Quick Filters**")

# Year filter if available
if 'Year' in df.columns:
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )

# --- 5. MAIN DASHBOARD CONTENT ---
st.markdown('<div class="main-header">Polio Vaccine Coverage Analysis</div>', unsafe_allow_html=True)

# Key metrics at the top
if 'Share polio vaccine (POL3)' in df.columns:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_coverage = df['Share polio vaccine (POL3)'].mean()
        st.metric("Average Coverage", f"{avg_coverage:.1f}%")
    
    with col2:
        max_coverage = df['Share polio vaccine (POL3)'].max()
        st.metric("Highest Coverage", f"{max_coverage:.1f}%")
    
    with col3:
        countries = df['Entity'].nunique() if 'Entity' in df.columns else "N/A"
        st.metric("Countries", countries)
    
    with col4:
        years_span = df['Year'].max() - df['Year'].min() if 'Year' in df.columns else "N/A"
        st.metric("Years Covered", years_span)

# --- 6. ANALYSIS SECTIONS ---

# DATA OVERVIEW
if analysis_type == "Data Overview":
    st.markdown('<div class="section-header">📈 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Key Information")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        st.write(f"**Data Period:** {df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A")
        
        if 'Share polio vaccine (POL3)' in df.columns:
            st.write(f"**Coverage Range:** {df['Share polio vaccine (POL3)'].min():.1f}% - {df['Share polio vaccine (POL3)'].max():.1f}%")

    # Basic statistics
    st.subheader("Statistical Summary")
    if 'Share polio vaccine (POL3)' in df.columns:
        st.dataframe(df['Share polio vaccine (POL3)'].describe(), use_container_width=True)

# VACCINE TRENDS
elif analysis_type == "Vaccine Trends":
    st.markdown('<div class="section-header">📊 Vaccine Trends Over Time</div>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in ['Entity', 'Year', 'Share polio vaccine (POL3)']):
        # Top countries selection
        top_countries = st.multiselect(
            "Select Countries to Display",
            options=df['Entity'].unique(),
            default=df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(5).index.tolist()
        )
        
        if top_countries:
            filtered_df = df[df['Entity'].isin(top_countries)]
            
            # Line chart
            fig = px.line(
                filtered_df, 
                x='Year', 
                y='Share polio vaccine (POL3)', 
                color='Entity',
                title='Polio Vaccine Coverage Over Time',
                labels={'Share polio vaccine (POL3)': 'Vaccine Coverage (%)', 'Year': 'Year'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution over time
        st.subheader("Coverage Distribution by Year")
        fig_box = px.box(
            df, 
            x='Year', 
            y='Share polio vaccine (POL3)',
            title='Vaccine Coverage Distribution by Year'
        )
        st.plotly_chart(fig_box, use_container_width=True)

# COUNTRY COMPARISON
elif analysis_type == "Country Comparison":
    st.markdown('<div class="section-header">🌍 Country Comparison</div>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in ['Entity', 'Share polio vaccine (POL3)']):
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performing countries
            st.subheader("Top Performing Countries")
            top_countries = df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(10)
            fig_bar = px.bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation='h',
                title='Average Vaccine Coverage by Country',
                labels={'x': 'Coverage (%)', 'y': 'Country'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Coverage distribution
            st.subheader("Global Coverage Distribution")
            fig_hist = px.histogram(
                df, 
                x='Share polio vaccine (POL3)',
                nbins=30,
                title='Distribution of Vaccine Coverage',
                labels={'Share polio vaccine (POL3)': 'Coverage (%)'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# STATISTICAL ANALYSIS
elif analysis_type == "Statistical Analysis":
    st.markdown('<div class="section-header">📐 Statistical Analysis</div>', unsafe_allow_html=True)
    
    if 'Share polio vaccine (POL3)' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Analysis")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Tests")
            
            # Basic stats
            coverage_data = df['Share polio vaccine (POL3)'].dropna()
            st.write(f"**Sample Size:** {len(coverage_data):,}")
            st.write(f"**Mean:** {coverage_data.mean():.2f}%")
            st.write(f"**Standard Deviation:** {coverage_data.std():.2f}%")
            st.write(f"**Variance:** {coverage_data.var():.2f}")
            
            # Normality test
            if len(coverage_data) > 3:
                stat, p_value = stats.normaltest(coverage_data)
                st.write(f"**Normality Test p-value:** {p_value:.4f}")

# --- 7. FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Polio Vaccine Dashboard • Built with Streamlit • Data Source: Your Dataset"
    "</div>", 
    unsafe_allow_html=True
)
