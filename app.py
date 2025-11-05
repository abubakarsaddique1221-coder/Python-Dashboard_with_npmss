import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import io

# --- 1. SET UP THE PAGE ---
# Use set_page_config for a professional, wide layout
st.set_page_config(
    page_title="Pro Data Explorer",
    page_icon="🔬",
    layout="wide"
)

# --- 2. CUSTOM CSS (To make it look cleaner) ---
# We can inject a little CSS to hide the "Made with Streamlit" footer
# and clean up the header spacing for a more "pro" look.
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Clean up the top margin */
            .block-container {
                padding-top: 2rem;
            }
            /* Style the sidebar */
            [data-testid="stSidebar"] {
                background-color: #0F172A; /* A deep blue/slate color */
            }
            [data-testid="stSidebar"] h2 {
                color: #FFFFFF;
            }
            [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 {
                color: #E2E8F0; /* Lighter text for sidebar */
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- 3. HELPER FUNCTION TO LOAD DATA ---
# This function will be cached, so Streamlit doesn't re-load the file
# every time you interact with a widget.
@st.cache_data
def load_data(source):
    """Loads data from file upload or URL into a pandas DataFrame."""
    try:
        if isinstance(source, io.StringIO) or isinstance(source, io.BytesIO):
            # From file uploader
            return pd.read_csv(source)
        elif isinstance(source, str):
            # From URL
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

# Section for uploading a file OR providing a URL
st.sidebar.header("1. Get Your Data")
data_source = st.sidebar.radio("Choose data source:", ("Upload a file", "Enter a URL"))

df = None  # Initialize df as None

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
    
    # Get column lists
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    # --- Sidebar - Analysis Options ---
    st.sidebar.header("2. Choose Your Analysis")
    
    # Check if we have the right column types for analysis
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

    # --- Render the selected analysis ---

    # -------------------------------
    # 1. DATA OVERVIEW
    # -------------------------------
    if analysis_type == "Data Overview":
        st.header("Data Overview")
        
        # Show raw data
        st.subheader("Raw Data (First 10 Rows)")
        st.dataframe(df.head(10))
        
        # Show descriptive statistics
        st.subheader("Descriptive Statistics (Numerical Columns)")
        if numeric_columns:
            st.dataframe(df[numeric_columns].describe())
        else:
            st.info("No numerical columns to describe.")
            
        # Show column information
        st.subheader("Column Information (Data Types & Non-Nulls)")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # -------------------------------
    # 2. UNIVARIATE ANALYSIS
    # -------------------------------
    elif analysis_type == "Univariate Analysis (1 Column)":
        st.header("Univariate Analysis")
        st.write("Explore the distribution of a single column.")
        
        selected_col = st.selectbox("Select a column:", all_columns)
        
        if selected_col:
            # Check if column is numeric or categorical
            if selected_col in numeric_columns:
                # --- Numeric Column ---
                st.subheader(f"Distribution of '{selected_col}' (Numeric)")
                
                # Plotly Histogram
                fig = px.histogram(
                    df, 
                    x=selected_col, 
                    marginal="box",  # Adds a box plot on top
                    title=f"Distribution of {selected_col}",
                    template="plotly_dark"  # Dark theme!
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("How to read this chart"):
                    st.write(f"""
                        This **histogram** shows the frequency distribution of the `{selected_col}` column.
                        - **X-axis:** The values of `{selected_col}`, grouped into bins (ranges).
                        - **Y-axis:** The count of rows that fall into each bin.
                        - The **box plot** at the top summarizes the distribution:
                            - The box represents the Interquartile Range (IQR) - the middle 50% of the data.
                            - The line in the box is the **median** (the 50th percentile).
                            - The "whiskers" extend to show the rest of the data, excluding outliers.
                    """)
            
            else:
                # --- Categorical Column ---
                st.subheader(f"Distribution of '{selected_col}' (Categorical)")
                
                # Get value counts
                counts = df[selected_col].value_counts().reset_index()
                counts.columns = [selected_col, 'count']
                
                # Plotly Bar Chart
                fig = px.bar(
                    counts.head(20),  # Show top 20 categories
                    x=selected_col, 
                    y='count',
                    title=f"Top 20 Value Counts for {selected_col}",
                    template="plotly_dark"
                )
                fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("How to read this chart"):
                    st.write(f"""
                        This **bar chart** shows the frequency of each category in the `{selected_col}` column.
                        - **X-axis:** The unique categories found in `{selected_col}` (limited to the top 20 most frequent).
                        - **Y-axis:** The count of how many times each category appears in the dataset.
                    """)

    # -------------------------------
    # 3. BIVARIATE ANALYSIS
    # -------------------------------
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

            # --- Case 1: Numeric vs. Numeric (Scatter Plot) ---
            if x_is_num and y_is_num:
                st.subheader(f"Relationship: '{x_col}' vs. '{y_col}' (Scatter Plot)")
                
                # Plotly Scatter Plot
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    trendline="ols",  # Adds an "Ordinary Least Squares" regression line
                    title=f"{x_col} vs. {y_col}",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("How to read this chart & analysis"):
                    st.write(f"""
                        This **scatter plot** shows the relationship between `{x_col}` and `{y_col}`.
                        - Each dot represents a single row in your data.
                        - **X-axis:** The value of `{x_col}`.
                        - **Y-axis:** The value of `{y_col}`.
                        - The **blue line** is a regression line that shows the general trend.
                            - **Upward slope:** Positive correlation (as X increases, Y tends to increase).
                            - **Downward slope:** Negative correlation (as X increases, Y tends to decrease).
                            - **Flat line:** No clear correlation.
                    """)
                
                # --- Inferential Statistics (Pearson Correlation) ---
                st.subheader("Inferential Statistics: Pearson Correlation")
                try:
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Pearson Correlation (r)", f"{corr:.3f}")
                    col2.metric("P-value", f"{p_value:.3g}") # 'g' for scientific notation if needed
                    
                    st.write(f"""
                        - **Correlation (r):** A value between -1 and 1.
                            - `r > 0`: Positive relationship.
                            - `r < 0`: Negative relationship.
                            - `r ≈ 0`: No linear relationship.
                            - The *closer to 1 or -1*, the *stronger* the relationship.
                        - **P-value:** Represents the probability that this correlation occurred by random chance.
                            - A **low p-value (typically < 0.05)** suggests the correlation is **statistically significant**.
                            - A **high p-value (typically > 0.05)** suggests the correlation is **not statistically significant** (it could be random).
                    """)
                    
                except Exception as e:
                    st.error(f"Could not calculate correlation: {e}")

            # --- Case 2: Categorical vs. Numeric (Box Plot) ---
            elif (not x_is_num and y_is_num) or (x_is_num and not y_is_num):
                cat_col = x_col if not x_is_num else y_col
                num_col = y_col if y_is_num else x_col
                
                st.subheader(f"Comparison: '{num_col}' by '{cat_col}' (Box Plot)")
                
                # Plotly Box Plot
                fig = px.box(
                    df, 
                    x=cat_col, 
                    y=num_col,
                    title=f"Distribution of {num_col} by {cat_col}",
                    template="plotly_dark",
                    points="all" # Show all the individual points
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("How to read this chart"):
                    st.write(f"""
                        This **box plot** compares the distribution of the numeric column `{num_col}` across different categories in `{cat_col}`.
                        - Each box represents a single category from `{cat_col}`.
                        - The **box** shows the Interquartile Range (IQR) - the middle 50% of the data for that category.
                        - The **line in the box** is the **median** for that category.
                        - **Whiskers** show the rest of the distribution.
                        - **Dots** are individual data points (sometimes outliers).
                        - **Use this to see:** Do different categories have different values? Is the median higher for one group? Is one group more spread out?
                    """)

            # --- Case 3: Categorical vs. Categorical (Heatmap) ---
            else:
                st.subheader(f"Relationship: '{x_col}' vs. '{y_col}' (Heatmap)")
                
                # Create a 2D frequency table
                contingency_table = pd.crosstab(df[x_col], df[y_col])
                
                # Plotly Heatmap
                fig = px.imshow(
                    contingency_table,
                    text_auto=True, # Show the counts in the squares
                    title=f"Frequency Heatmap: {x_col} vs. {y_col}",
                    template="plotly_dark",
                    color_continuous_scale="Viridis" # A nice gradient
                )
                fig.update_layout(xaxis_title=y_col, yaxis_title=x_col)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("How to read this chart"):
                    st.write(f"""
                        This **heatmap** shows the frequency of co-occurrence between categories from `{x_col}` and `{y_col}`.
                        - **X-axis:** Categories from `{y_col}`.
                        - **Y-axis:** Categories from `{x_col}`.
                        - **Color & Number:** The number in each square shows how many rows have that specific *combination* of categories.
                        - The **color intensity** (from dark blue to yellow) also represents the count.
                        - **Use this to see:** Do certain categories from `{x_col}` frequently appear with categories from `{y_col}`?
                    """)

    # -------------------------------
    # 4. MULTIVARIATE ANALYSIS
    # -------------------------------
    elif analysis_type == "Multivariate Analysis (Heatmap)":
        st.header("Multivariate Analysis: Correlation Heatmap")
        st.write("Explore the linear relationships between all numeric columns.")
        
        if len(numeric_columns) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numeric_columns].corr()
            
            # Plotly Heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,  # Show correlation values
                aspect="auto",
                title="Correlation Heatmap of Numeric Columns",
                template="plotly_dark",
                color_continuous_scale="RdBu_r", # Red-Blue diverging scale
                zmin=-1, zmax=1  # Fix the scale from -1 to 1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to read this chart"):
                st.write(f"""
                    This **correlation heatmap** shows the Pearson Correlation (r) between every pair of numeric columns.
                    - **Color Scale (from -1 to 1):**
                        - **Strong Red (≈ 1.0):** Strong positive correlation (as one increases, the other increases).
                        - **Strong Blue (≈ -1.0):** Strong negative correlation (as one increases, the other decreases).
                        - **White/Light (≈ 0):** No linear correlation.
                    - The diagonal is always 1.0 because a variable is perfectly correlated with itself.
                """)
        else:
            st.info("You need at least two numeric columns in your data to create a correlation heatmap.")