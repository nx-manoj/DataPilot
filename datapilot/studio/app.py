import streamlit as st
import pandas as pd
import polars as pl
import io
import os
import sys

# Add the parent directory to sys.path so we can import datapilot from the source if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import datapilot as dp
from datapilot.config import configure, get_config

st.set_page_config(
    page_title="DataPilot Web Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 DataPilot Web Studio")
st.markdown("Automated high-speed profiling and Conversational AI.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Provider Settings
    st.subheader("AI Settings")
    ai_provider = st.selectbox("AI Provider", ["ollama", "openai", "gemini", "claude", "groq"], index=0)
    
    api_key = ""
    if ai_provider != "ollama":
        api_key = st.text_input("API Key", type="password")
        
    ai_model = st.text_input("Model (leave blank for default)", value="")
    
    if st.button("Apply Settings"):
        configure(
            ai_provider=ai_provider,
            ai_model=ai_model if ai_model else None,
            api_key=api_key if api_key else None
        )
        st.success(f"Configured with {ai_provider}!")

    st.divider()
    
    # File Uploader
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    # Load Data
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
        
    # Store df in session state for chat functionality
    st.session_state["df"] = df
    
    # Run analysis
    meta = dp.summary(df)
    
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Chat Copilot", "Data Preview"])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        # Top-level metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Total Rows</div><div class="metric-value">{meta["rows"]:,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Columns</div><div class="metric-value">{meta["columns"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Engine</div><div class="metric-value" style="font-size:22px">{meta["engine_detected"].upper()}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Memory</div><div class="metric-value">{meta["memory_usage_mb"]} MB</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Missing Values</div><div class="metric-value">{meta["total_missing_values"]:,}</div></div>', unsafe_allow_html=True)
            
        st.divider()
        
        # Missing values and Duplicates
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Missing Values")
            missing_df = dp.missing(df)
            st.dataframe(missing_df, use_container_width=True)
            
        with col_right:
            st.subheader("Data Types")
            dtypes = pd.DataFrame([{"Column": k, "Type": str(v)} for k, v in df.dtypes.items()])
            st.dataframe(dtypes, use_container_width=True)
            
        st.divider()
        st.subheader("Correlation Heatmap")
        with st.spinner("Generating heatmap..."):
            fig = dp.heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns for a correlation heatmap.")
                
        st.divider()
        st.subheader("Distribution Overviews")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize:", numeric_cols)
            with st.spinner("Generating histogram..."):
                hist_fig = dp.hist(df, selected_col)
                st.plotly_chart(hist_fig, use_container_width=True)
                
    # --- TAB 2: CHAT COPILOT ---
    with tab2:
        st.markdown("""
        ### Ask your AI Copilot
        Type a plain-English prompt like **"Show the relation between Age and Fare"** or **"What are the most important preprocessing steps?"**
        """)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "figure" in message:
                    st.plotly_chart(message["figure"], use_container_width=True)
                else:
                    st.markdown(message["content"])
                    
        # Chat input
        if prompt := st.chat_input("Plot the distribution of..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing request..."):
                    # First try to see if it's a visualization request
                    # We can use a heuristic or just try visualize_ai
                    # To be smart, let's look for plot keywords
                    plot_keywords = ["plot", "show", "draw", "graph", "chart", "visualize", "distribution", "correlation", "heatmap", "relation", "compare"]
                    
                    is_plot_request = any(kw in prompt.lower() for kw in plot_keywords)
                    
                    if is_plot_request:
                        fig = dp.visualize_ai(df, prompt)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "content": "Here is the chart you requested.", "figure": fig})
                        else:
                            st.markdown("I couldn't determine a valid chart for that request. Trying text analysis...")
                            response = dp.ask_ai(df, prompt)
                            # ask_ai returns a _SilentStr wrapper which prints out but also gives string
                            st.markdown(str(response))
                            st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    else:
                        response = dp.ask_ai(df, prompt)
                        st.markdown(str(response))
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})

    # --- TAB 3: DATA PREVIEW ---
    with tab3:
        st.dataframe(df.head(100), use_container_width=True)
else:
    # Empty State
    st.info("👈 Please upload a CSV file from the sidebar to get started.")
