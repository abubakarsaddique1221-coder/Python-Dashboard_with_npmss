import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats  # <-- THIS LINE NEEDS SCIPY
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

# --- 4. SIDEBAR - CONTROLS ---
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

# --- 5. MAIN PAGE - ANALYSIS ---
st.title("Pro Data Explorer")

if df is None:
    st.info("Please upload a CSV file or provide a URL in the sidebar to begin.")
else:
    st.success("Data loaded successfully!")
    
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    st.sidebar.header("2. Choose Your Analysis")
    
    if not numeric_columns:
        st.sidebar.warning("No numeric columns found for some analyses.")
    if not categorical_columns:
        st.sidebar.warning("No categorical columns found for some analyses.")

    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        [
            "Data Overview",
            "Univariate Analysis (1 Column)",
            "Bivariate Analysis (2 Columns)",
            "Multivariate Analysis (Heatmap)"
        ]
    )

    if analysis_type == "Data Overview":
        st.header("Data Overview")
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
        s = buffer.getvalue()
        st.text(s)

    elif analysis_type == "Univariate Analysis (1 Column)":
        st.header("Univariate Analysis")
        st.write("Explore the distribution of a single column.")
        
        selected_col = st.selectbox("Select a column:", all_columns)
        
        if selected_col:
            if selected_col in numeric_columns:
                st.subheader(f"Distribution of '{selected_col}' (Numeric)")
                fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution of {selected_col}", template="plotly_dark")
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("How to read this chart"):
                    st.write(f"""...""") # Explanation
            else:
                st.subheader(f"Distribution of '{selected_col}' (Categorical)")
                counts = df[selected_col].value_counts().reset_index()
                counts.columns = [selected_col, 'count']
                fig = px.bar(counts.head(20), x=selected_col, y='count', title=f"Top 20 Value Counts for {selected_col}", template="plotly_dark")
                fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("How to read this chart"):
                    st.write(f"""...""") # Explanation

    elif analysis_type == "Bivariate Analysis (2 Columns)":
        st.header("Bivariate Analysis")
        st.write("Explore the relationship between two columns.")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column:", all_columns)
        with col2:
            y_col = st.selectbox("Select Y-axis column:", all_columns)

        if x_col and y_col:
            x_is_num = x_col in numeric_columns
            y_is_num = y_col in numeric_columns

            if x_is_num and y_is_num:
                st.subheader(f"Relationship: '{x_col}' vs. '{y_col}' (Scatter Plot)")
                fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs. {y_col}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("How to read this chart & analysis"):
                    st.write(f"""...""") # Explanation
                
                st.subheader("Inferential Statistics: Pearson Correlation")
                try:
                    corr, p_value = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna()) # <-- THIS LINE NEEDS SCIPY
                    col1, col2 = st.columns(2)
                    col1.metric("Pearson Correlation (r)", f"{corr:.3f}")
                    col2.metric("P-value", f"{p_value:.3g}")
                    st.write(f"""...""") # Explanation
                except Exception as e:
                    st.error(f"Could not calculate correlation: {e}")

            elif (not x_is_num and y_is_num) or (x_is_num and not y_is_num):
                cat_col = x_col if not x_is_num else y_col
                num_col = y_col if y_is_num else x_col
                st.subheader(f"Comparison: '{num_col}' by '{cat_col}' (Box Plot)")
                fig = px.box(df, x=cat_col, y=num_col, title=f"Distribution of {num_col} by {cat_col}", template="plotly_dark", points="all")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("How to read this chart"):
                    st.write(f"""...""") # Explanation

            else:
                st.subheader(f"Relationship: '{x_col}' vs. '{y_col}' (Heatmap)")
                contingency_table = pd.crosstab(df[x_col], df[y_col])
                fig = px.imshow(contingency_table, text_auto=True, title=f"Frequency Heatmap: {x_col} vs. {y_col}", template="plotly_dark", color_continuous_scale="Viridis")
                fig.update_layout(xaxis_title=y_col, yaxis_title=x_col)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("How to read this chart"):
                    st.write(f"""...""") # Explanation

    elif analysis_type == "Multivariate Analysis (Heatmap)":
        st.header("Multivariate Analysis: Correlation Heatmap")
        st.write("Explore the linear relationships between all numeric columns.")
        
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap of Numeric Columns", template="plotly_dark", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("How to read this chart"):
                st.write(f"""...""") # Explanation
        else:
            st.info("You need at least two numeric columns in your data to create a correlation heatmap.")
