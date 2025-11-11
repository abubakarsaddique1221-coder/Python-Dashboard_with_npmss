import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import time
from datetime import datetime
import io

# --- 1. SECURITY CONFIGURATION ---
SESSION_TIMEOUT = 1800  # 30 minutes

# Simple user database
USER_CREDENTIALS = {
    "admin": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
        "role": "admin",
        "name": "System Administrator"
    },
    "analyst": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
        "role": "analyst",
        "name": "Data Analyst"
    },
    "viewer": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
        "role": "viewer",
        "name": "Viewer"
    }
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    if username in USER_CREDENTIALS:
        hashed_password = hash_password(password)
        return USER_CREDENTIALS[username]["password"] == hashed_password
    return False

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = time.time()

def check_session_timeout():
    if st.session_state.authenticated:
        current_time = time.time()
        if current_time - st.session_state.last_activity > SESSION_TIMEOUT:
            logout()
            st.error("Session timed out due to inactivity. Please log in again.")
            st.stop()
        st.session_state.last_activity = current_time

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- 2. PROFESSIONAL STYLING ---
def apply_custom_styling():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2e86ab;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem 0;
            font-weight: 600;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .sidebar-header {
            font-size: 1.2rem;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .user-info {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.8rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
        .dataframe {
            border-radius: 10px;
        }
        .logout-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION COMPONENT ---
def login_component():
    st.markdown('<div class="main-header">🔐 Polio Vaccine Analytics Portal</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### Secure Login")
            
            with st.form("login_form"):
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("🚀 Login")
                
                if submit:
                    if check_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = USER_CREDENTIALS[username]["role"]
                        st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.last_activity = time.time()
                        st.success(f"Welcome back, {USER_CREDENTIALS[username]['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            st.markdown("---")
            st.markdown("**Demo Credentials:**")
            st.info("""
            - **Admin**: admin / password
            - **Analyst**: analyst / password  
            - **Viewer**: viewer / password
            """)

# --- 4. DATA MANAGEMENT ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Try without index_col first
        df = pd.read_csv('new cleaned file (1).csv')
        
        # If first column is index, set it properly
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        
        # Data validation and cleaning
        if 'Share polio vaccine (POL3)' in df.columns:
            df['Share polio vaccine (POL3)'] = pd.to_numeric(
                df['Share polio vaccine (POL3)'], errors='coerce'
            )
        
        return df
    except Exception as e:
        st.error(f"🚨 Data loading error: {str(e)}")
        return None

def get_user_permissions(role):
    permissions = {
        "admin": {"view": True, "analyze": True, "export": True, "manage": True},
        "analyst": {"view": True, "analyze": True, "export": True, "manage": False},
        "viewer": {"view": True, "analyze": False, "export": False, "manage": False}
    }
    return permissions.get(role, permissions["viewer"])

# --- 5. DASHBOARD COMPONENTS ---
def render_sidebar(df):
    with st.sidebar:
        st.markdown(f'<div class="sidebar-header">💉 Polio Vaccine Dashboard</div>', unsafe_allow_html=True)
        
        # User info card
        st.markdown(f"""
        <div class="user-info">
            👤 <strong>{st.session_state.username}</strong><br/>
            🎯 Role: {st.session_state.role}<br/>
            ⏰ Login: {st.session_state.login_time}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📊 Navigation")
        
        if st.session_state.role in ["admin", "analyst"]:
            analysis_options = [
                "Dashboard Overview",
                "Vaccine Trends", 
                "Country Analysis",
                "Statistical Reports",
                "Data Export"
            ]
        else:
            analysis_options = ["Dashboard Overview", "Country Analysis"]
        
        selected_analysis = st.radio("Select Analysis", analysis_options)
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        if 'Year' in df.columns:
            years = sorted(df['Year'].unique())
            selected_years = st.slider(
                "Select Year Range",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
        
        if 'Entity' in df.columns:
            entities = st.multiselect(
                "Select Countries/Entities",
                options=df['Entity'].unique(),
                default=df['Entity'].value_counts().head(5).index.tolist()
            )
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", key="logout", use_container_width=True):
            logout()
    
    return selected_analysis

def render_dashboard_overview(df):
    st.markdown('<div class="main-header">📈 Polio Vaccine Coverage Dashboard</div>', unsafe_allow_html=True)
    
    # Key Performance Indicators
    st.markdown("### 🎯 Key Performance Indicators")
    
    if 'Share polio vaccine (POL3)' in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_coverage = df['Share polio vaccine (POL3)'].mean()
            st.metric("Global Average Coverage", f"{avg_coverage:.1f}%")
        
        with col2:
            max_coverage = df['Share polio vaccine (POL3)'].max()
            st.metric("Highest Coverage", f"{max_coverage:.1f}%")
        
        with col3:
            countries = df['Entity'].nunique() if 'Entity' in df.columns else 0
            st.metric("Countries Covered", f"{countries:,}")
        
        with col4:
            years_span = f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A"
            st.metric("Data Period", years_span)
    
    # Quick Insights
    st.markdown("### 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ['Entity', 'Share polio vaccine (POL3)']):
            st.subheader("🏆 Top 10 Performing Countries")
            top_countries = df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(10)
            fig_bar = px.bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation='h',
                title='',
                labels={'x': 'Coverage (%)', 'y': ''},
                color=top_countries.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        if 'Share polio vaccine (POL3)' in df.columns:
            st.subheader("📊 Coverage Distribution")
            fig_hist = px.histogram(
                df, 
                x='Share polio vaccine (POL3)',
                nbins=30,
                title='',
                labels={'Share polio vaccine (POL3)': 'Coverage (%)'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def render_vaccine_trends(df):
    st.markdown('<div class="section-header">📈 Vaccine Coverage Trends</div>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in ['Entity', 'Year', 'Share polio vaccine (POL3)']):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Settings")
            selected_countries = st.multiselect(
                "Select Countries",
                options=df['Entity'].unique(),
                default=df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(5).index.tolist()
            )
            
            chart_type = st.radio("Chart Type", ["Line Chart", "Area Chart"])
        
        with col1:
            if selected_countries:
                filtered_df = df[df['Entity'].isin(selected_countries)]
                
                if chart_type == "Line Chart":
                    fig = px.line(
                        filtered_df, 
                        x='Year', 
                        y='Share polio vaccine (POL3)', 
                        color='Entity',
                        title='Polio Vaccine Coverage Trends',
                        labels={'Share polio vaccine (POL3)': 'Coverage (%)'}
                    )
                else:
                    fig = px.area(
                        filtered_df, 
                        x='Year', 
                        y='Share polio vaccine (POL3)', 
                        color='Entity',
                        title='Polio Vaccine Coverage Trends',
                        labels={'Share polio vaccine (POL3)': 'Coverage (%)'}
                    )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one country to display trends.")

def render_country_analysis(df):
    st.markdown('<div class="section-header">🌍 Country Analysis</div>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in ['Entity', 'Share polio vaccine (POL3)']):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Country Performance Matrix")
            
            # Performance categories
            performance_df = df.groupby('Entity')['Share polio vaccine (POL3)'].agg(['mean', 'std']).round(2)
            performance_df['Performance'] = pd.cut(
                performance_df['mean'], 
                bins=[0, 50, 80, 90, 100],
                labels=['Low', 'Medium', 'High', 'Excellent']
            )
            
            st.dataframe(performance_df, use_container_width=True)
        
        with col2:
            st.subheader("Regional Comparison")
            
            # Top countries pie chart
            top_10 = df.groupby('Entity')['Share polio vaccine (POL3)'].mean().nlargest(10)
            fig = px.pie(
                values=top_10.values,
                names=top_10.index,
                title='Top 10 Countries by Coverage'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_statistical_reports(df):
    st.markdown('<div class="section-header">📊 Statistical Reports</div>', unsafe_allow_html=True)
    
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
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Need more numeric columns for correlation analysis.")
        
        with col2:
            st.subheader("Statistical Summary")
            
            coverage_data = df['Share polio vaccine (POL3)'].dropna()
            
            stats_data = {
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    len(coverage_data),
                    f"{coverage_data.mean():.2f}%",
                    f"{coverage_data.std():.2f}%",
                    f"{coverage_data.min():.1f}%",
                    f"{coverage_data.quantile(0.25):.1f}%",
                    f"{coverage_data.median():.1f}%",
                    f"{coverage_data.quantile(0.75):.1f}%",
                    f"{coverage_data.max():.1f}%"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

def render_data_export(df):
    st.markdown('<div class="section-header">📁 Data Export</div>', unsafe_allow_html=True)
    
    st.subheader("Export Data")
    
    export_format = st.radio("Export Format", ["CSV", "Excel"])
    
    if st.button("📥 Generate Export"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"polio_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Polio Data')
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"polio_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

# --- 6. MAIN APPLICATION ---
def main():
    # Initialize and check session
    initialize_session_state()
    apply_custom_styling()
    
    # Check authentication
    if not st.session_state.authenticated:
        login_component()
        return
    
    # Check session timeout
    check_session_timeout()
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Unable to load data. Please contact administrator.")
        return
    
    # Get user permissions
    permissions = get_user_permissions(st.session_state.role)
    
    # Render sidebar and get selected analysis
    selected_analysis = render_sidebar(df)
    
    # Render main content based on selection and permissions
    if selected_analysis == "Dashboard Overview":
        render_dashboard_overview(df)
    elif selected_analysis == "Vaccine Trends" and permissions["analyze"]:
        render_vaccine_trends(df)
    elif selected_analysis == "Country Analysis":
        render_country_analysis(df)
    elif selected_analysis == "Statistical Reports" and permissions["analyze"]:
        render_statistical_reports(df)
    elif selected_analysis == "Data Export" and permissions["export"]:
        render_data_export(df)
    else:
        st.warning("⚠️ You don't have permission to access this section.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Polio Vaccine Analytics Portal • Secure Access • © 2024"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
