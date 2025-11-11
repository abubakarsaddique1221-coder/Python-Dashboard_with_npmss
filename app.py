import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io

# --- 1. SET UP THE PAGE ---
st.set_page_config(
    page_title="Pro Data Explorer",
    page_icon="🔬",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] { background-color: #0F172A; }
    [data-testid="stSidebar"] h2 { color: #FFFFFF; }
    [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 { color: #E2E8F0; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 3. HELPER FUNCTION TO LOAD DATA ---
@st.cache_data
def load_data(source):
    try:
        if isinstance(source, io.StringIO) or isinstance(source, io.BytesIO):
            return pd.read_csv(source)
        elif isinstance(source, str):
            if source.endswith('.csv'):
                return pd.read_csv(source)
            else:
                st.error("URL must point to a .csv file.")
                return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return None

# --- 4. GRAPH CUSTOMIZATION OPTIONS ---
def get_color_themes():
    return {
        "Plotly Dark": "plotly_dark",
        "Plotly White": "plotly_white",
        "Plotly": "plotly",
        "GGPlot": "ggplot2",
        "Seaborn": "seaborn",
        "Simple White": "simple_white",
        "None (Custom Colors)": None
    }

def get_color_scales():
    return [
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Blues", "Reds", "Greens", "Purples", "Oranges",
        "Rainbow", "Portland", "Hot", "Cool", "RdBu", "RdBu_r",
        "Picnic", "Jet", "Electric"
    ]

def get_marker_symbols():
    return ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]

# --- 5. SIDEBAR - CONTROLS ---
st.sidebar.title("Data Explorer 🔬")
st.sidebar.write("Upload your data and start exploring!")
st.sidebar.header("1. Get Your Data")
data_source = st.sidebar.radio("Choose data source:", ("Upload a file", "Enter a URL"))

df = None

if data_source == "Upload a file":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
else:
    data_url = st.sidebar.text_input("Paste a URL to a CSV file")
    if data_url:
        df = load_data(data_url)

# --- 6. MAIN PAGE - ANALYSIS ---
st.title("Pro Data Explorer")

if df is None:
    st.info("Please upload a CSV file or provide a URL in the sidebar to begin.")
else:
    st.success("✅ Data loaded successfully!")
    
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    st.sidebar.header("2. Choose Your Analysis")
    
    if not numeric_columns:
        st.sidebar.warning("⚠️ No numeric columns found for some analyses.")
    if not categorical_columns:
        st.sidebar.warning("⚠️ No categorical columns found for some analyses.")

    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        [
            "Data Overview",
            "Univariate Analysis (1 Column)",
            "Bivariate Analysis (2 Columns)",
            "Multivariate Analysis (Heatmap)",
            "Advanced Visualizations"
        ]
    )

    # Graph customization section in sidebar
    st.sidebar.header("3. Graph Customization")
    
    color_themes = get_color_themes()
    selected_theme = st.sidebar.selectbox("Color Theme", list(color_themes.keys()))
    template = color_themes[selected_theme]
    
    # Show additional customization options based on analysis type
    show_customization = st.sidebar.checkbox("Show Advanced Graph Options", value=False)
    
    if show_customization:
        color_scale = st.sidebar.selectbox("Color Scale", get_color_scales(), index=0)
        marker_symbol = st.sidebar.selectbox("Marker Symbol", get_marker_symbols(), index=0)
        marker_size = st.sidebar.slider("Marker Size", 1, 20, 6)
        line_width = st.sidebar.slider("Line Width", 1, 10, 2)
        opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.8)
        show_grid = st.sidebar.checkbox("Show Grid", value=True)
    else:
        color_scale = "Viridis"
        marker_symbol = "circle"
        marker_size = 6
        line_width = 2
        opacity = 0.8
        show_grid = True

    # ----------- ANALYSIS SECTIONS -----------

    # DATA OVERVIEW
    if analysis_type == "Data Overview":
        st.header("📊 Data Overview")
        st.subheader("Raw Data (First 10 Rows)")
        st.dataframe(df.head(10))
        
        st.subheader("Descriptive Statistics (Numerical Columns)")
        if numeric_columns:
            st.dataframe(df[numeric_columns].describe())
        else:
            st.info("No numerical columns to describe.")
            
        st.subheader("Column Information (Data Types & Non-Nulls)")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Data Quality Check")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

    # UNIVARIATE ANALYSIS
    elif analysis_type == "Univariate Analysis (1 Column)":
        st.header("📈 Univariate Analysis")
        selected_col = st.selectbox("Select a column:", all_columns)
        
        if selected_col:
            if selected_col in numeric_columns:
                chart_type = st.radio("Select Chart Type:", 
                                      ["Histogram", "Box Plot", "Violin Plot", "Density Plot"], 
                                      horizontal=True)
                
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=selected_col, marginal="box", 
                                     title=f"Distribution of {selected_col}", 
                                     template=template,
                                     color_discrete_sequence=px.colors.sequential[color_scale])
                elif chart_type == "Box Plot":
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}", 
                                 template=template,
                                 color_discrete_sequence=px.colors.sequential[color_scale])
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, y=selected_col, title=f"Violin Plot of {selected_col}", 
                                    template=template,
                                    color_discrete_sequence=px.colors.sequential[color_scale])
                else:
                    fig = px.histogram(df, x=selected_col, marginal="rug",
                                     title=f"Density Plot of {selected_col}", 
                                     template=template,
                                     color_discrete_sequence=px.colors.sequential[color_scale])

                # ✅ Fixed showgrid
                fig.update_xaxes(showgrid=show_grid)
                fig.update_yaxes(showgrid=show_grid)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                chart_type = st.radio("Select Chart Type:", ["Bar Chart", "Pie Chart", "Treemap"], horizontal=True)
                counts = df[selected_col].value_counts().reset_index()
                counts.columns = [selected_col, 'count']
                
                if chart_type == "Bar Chart":
                    fig = px.bar(counts.head(20), x=selected_col, y='count', 
                               title=f"Top 20 Values for {selected_col}", 
                               template=template,
                               color='count', color_continuous_scale=color_scale)
                elif chart_type == "Pie Chart":
                    fig = px.pie(counts.head(10), names=selected_col, values='count',
                               title=f"Pie Chart of {selected_col}", template=template)
                else:
                    fig = px.treemap(counts.head(15), path=[selected_col], values='count',
                                   title=f"Treemap of {selected_col}", template=template,
                                   color='count', color_continuous_scale=color_scale)
                
                fig.update_xaxes(showgrid=show_grid)
                fig.update_yaxes(showgrid=show_grid)
                st.plotly_chart(fig, use_container_width=True)

    # BIVARIATE ANALYSIS
    elif analysis_type == "Bivariate Analysis (2 Columns)":
        st.header("🔍 Bivariate Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", all_columns)
        with col2:
            y_col = st.selectbox("Select Y-axis column:", all_columns)

        if x_col and y_col:
            x_is_num = x_col in numeric_columns
            y_is_num = y_col in numeric_columns

            # NUMERIC vs NUMERIC
            if x_is_num and y_is_num:
                chart_type = st.radio("Chart Type:", ["Scatter Plot", "Line Plot", "Hexbin Plot"], horizontal=True)
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                   title=f"{x_col} vs. {y_col}", 
                                   template=template, opacity=opacity,
                                   color_discrete_sequence=px.colors.sequential[color_scale])
                    fig.update_traces(marker=dict(size=marker_size, symbol=marker_symbol))
                elif chart_type == "Line Plot":
                    fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs. {y_col}", template=template)
                    fig.update_traces(line=dict(width=line_width))
                else:
                    fig = px.density_heatmap(df, x=x_col, y=y_col, 
                                           title=f"Hexbin Plot: {x_col} vs. {y_col}",
                                           template=template, color_continuous_scale=color_scale)

                fig.update_xaxes(showgrid=show_grid)
                fig.update_yaxes(showgrid=show_grid)
                st.plotly_chart(fig, use_container_width=True)

            # CATEGORICAL vs NUMERIC
            elif (not x_is_num and y_is_num) or (x_is_num and not y_is_num):
                cat_col = x_col if not x_is_num else y_col
                num_col = y_col if y_is_num else x_col
                chart_type = st.radio("Chart Type:", ["Box Plot", "Violin Plot", "Bar Chart", "Strip Plot"], horizontal=True)
                if chart_type == "Box Plot":
                    fig = px.box(df, x=cat_col, y=num_col, template=template, color=cat_col)
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=cat_col, y=num_col, template=template, color=cat_col)
                elif chart_type == "Bar Chart":
                    agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(agg_df, x=cat_col, y=num_col, template=template, color=cat_col)
                else:
                    fig = px.strip(df, x=cat_col, y=num_col, template=template, color=cat_col)

                fig.update_xaxes(showgrid=show_grid)
                fig.update_yaxes(showgrid=show_grid)
                st.plotly_chart(fig, use_container_width=True)

    # MULTIVARIATE (HEATMAP)
    elif analysis_type == "Multivariate Analysis (Heatmap)":
        st.header("🔥 Multivariate Analysis: Correlation Heatmap")
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap",
                            template=template, color_continuous_scale=color_scale)
            fig.update_xaxes(showgrid=show_grid)
            fig.update_yaxes(showgrid=show_grid)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("You need at least two numeric columns for this analysis.")

# --- 7. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("📊 Pro Data Explorer v1.1 — Explore your datasets visually and interactively!")
