"""
AI Data Analyst - Streamlit Application
========================================
A simple AI-powered application that allows users to upload CSV datasets,
view statistics, generate charts, and receive AI-driven insights using
the Google Gemini API.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS for premium styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit deploy button and menu */
    .stDeployButton, #MainMenu {display: none !important;}

    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(27, 94, 32, 0.4);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Card-like sections */
    .section-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
        transition: box-shadow 0.3s ease;
    }
    .section-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 5px solid #667eea;
    }

    /* Statistics highlight */
    .stat-highlight {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        color: #1a1a2e !important;
    }
    .chat-message strong {
        color: #1a1a2e !important;
    }
    .chat-user {
        background: #e8eaf6;
        border-left: 4px solid #667eea;
    }
    .chat-ai {
        background: #f3e5f5;
        border-left: 4px solid #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 AI Data Analyst</h1>
    <p>Upload a CSV dataset and let AI uncover insights, trends, and recommendations for you.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar – API Key Configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")

    # Allow the user to input their Groq API key (or read from env)
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at https://console.groq.com/keys",
    )

    # Model selection
    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
        help="Select the Groq model to use.",
    )

    st.markdown("---")
    st.markdown(
        f"Built with ❤️ using **Streamlit**, **Pandas**, **Plotly**, and **Groq**."
    )


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def build_dataset_summary(df: pd.DataFrame) -> str:
    """
    Build a textual summary of the dataset to send to the LLM.
    Includes shape, column info, basic statistics, and sample rows.
    """
    summary_parts = []

    # Basic shape
    summary_parts.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Column names and types
    summary_parts.append("Columns and data types:")
    for col in df.columns:
        summary_parts.append(f"  - {col}: {df[col].dtype}")
    summary_parts.append("")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        summary_parts.append("Missing values per column:")
        for col, count in missing.items():
            if count > 0:
                summary_parts.append(f"  - {col}: {count} ({count / len(df) * 100:.1f}%)")
        summary_parts.append("")

    # Descriptive statistics (for numeric columns)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary_parts.append("Descriptive statistics (numeric columns):")
        summary_parts.append(df[numeric_cols].describe().to_string())
        summary_parts.append("")

    # Categorical column info
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        summary_parts.append("Categorical columns – unique values and top entries:")
        for col in cat_cols:
            unique_count = df[col].nunique()
            top_values = df[col].value_counts().head(5).to_dict()
            summary_parts.append(f"  - {col}: {unique_count} unique → {top_values}")
        summary_parts.append("")

    # First few rows as a sample
    summary_parts.append("Sample rows (first 5):")
    summary_parts.append(df.head().to_string())

    return "\n".join(summary_parts)




def get_groq_client(api_key: str):
    """Create and return a Groq client."""
    return Groq(api_key=api_key)


def call_with_retry(func, max_retries=3):
    """
    Call a function with automatic retry on rate limit (429) errors.
    Shows a friendly countdown in the Streamlit UI while waiting.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 60  # Wait 60 seconds each retry
                if attempt < max_retries - 1:
                    countdown = st.empty()
                    for remaining in range(wait_time, 0, -1):
                        countdown.info(
                            f"⏳ Rate limit hit. Retrying in {remaining}s... "
                            f"(attempt {attempt + 2}/{max_retries})"
                        )
                        time.sleep(1)
                    countdown.empty()
                else:
                    raise Exception(
                        "Rate limit exceeded after multiple retries. "
                        "Please wait a few minutes and try again, or create an API key in a NEW project at https://aistudio.google.com/app/apikey"
                    )
            else:
                raise


def get_ai_insights(summary: str, api_key: str, model: str) -> str:
    """
    Send the dataset summary to the Groq API and return AI-generated insights.
    """
    prompt = (
        "You are an expert data analyst. Below is a summary of a dataset. "
        "Analyze it thoroughly and provide:\n"
        "1. **Key Insights** – the most important findings from the data.\n"
        "2. **Trends** – any noticeable patterns or trends.\n"
        "3. **Anomalies** – unusual data points or outliers.\n"
        "4. **Business Recommendations** – actionable suggestions based on the analysis.\n\n"
        "Be concise, clear, and use markdown formatting for readability.\n\n"
        f"Here is the dataset summary:\n\n{summary}"
    )

    client = get_groq_client(api_key)
    def _call():
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return response.choices[0].message.content
    return call_with_retry(_call)


def get_chat_response(chat_history: list, user_message: str, dataset_summary: str, api_key: str, model: str) -> str:
    """
    Send a chat conversation to the Groq API and return the response.
    """
    system_instruction = (
        "You are an expert data analyst assistant. The user has uploaded a dataset. "
        f"Here is the dataset summary:\n\n{dataset_summary}\n\n"
        "Answer questions about this data accurately. Use markdown formatting. "
        "If asked for calculations, explain your reasoning. "
        "If a question cannot be answered from the available data, say so clearly."
    )

    client = get_groq_client(api_key)
    messages = [{"role": "system", "content": system_instruction}]
    for msg in chat_history:
        messages.append({"role": msg["role"] if msg["role"] != "model" else "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    def _call():
        response = client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return response.choices[0].message.content
    return call_with_retry(_call)


# ──────────────────────────────────────────────
# File Uploader
# ──────────────────────────────────────────────
st.subheader("📁 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file to begin analysis. A sample dataset (sample_data.csv) is included in the project.",
)

# ──────────────────────────────────────────────
# Main Analysis Pipeline
# ──────────────────────────────────────────────
if uploaded_file is not None:
    # Load the CSV into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # ── Dataset Preview ──────────────────────
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Show quick stats in metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Numeric Cols", f"{len(df.select_dtypes(include='number').columns):,}")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    # ══════════════════════════════════════════
    # Tabbed Analysis Sections
    # ══════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Statistics",
        "📊 Visualizations",
        "🔗 Correlations",
        "🔎 Data Profiling",
        "🤖 AI Insights",
        "💬 Chat with Data",
    ])

    # ── Tab 1: Basic Statistics ──────────────
    with tab1:
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # Missing data visualization
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0]

        if not missing_df.empty:
            st.markdown("#### ⚠️ Missing Data")
            fig_missing = px.bar(
                missing_df,
                x="Column",
                y="Missing %",
                color="Missing %",
                color_continuous_scale="Reds",
                title="Missing Data by Column",
                text="Missing Count",
            )
            fig_missing.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                showlegend=False,
            )
            fig_missing.update_traces(textposition="outside")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing data found in the dataset!")

    # ── Tab 2: Visualizations ────────────────
    with tab2:
        st.subheader("📊 Visualizations")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            # Column selector
            selected_cols = st.multiselect(
                "Select numeric columns to chart",
                options=numeric_cols,
                default=numeric_cols[:3],
            )

            if selected_cols:
                # Chart type selector
                chart_type = st.radio(
                    "Chart type",
                    ["Line Chart", "Bar Chart", "Area Chart"],
                    horizontal=True,
                )

                if chart_type == "Line Chart":
                    fig = px.line(
                        df, y=selected_cols,
                        title="Line Chart",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                elif chart_type == "Bar Chart":
                    fig = px.bar(
                        df, y=selected_cols,
                        title="Bar Chart",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                else:
                    fig = px.area(
                        df, y=selected_cols,
                        title="Area Chart",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Histograms
                st.markdown("#### Distributions")
                hist_cols_layout = st.columns(min(len(selected_cols), 3))
                for i, col in enumerate(selected_cols):
                    with hist_cols_layout[i % len(hist_cols_layout)]:
                        fig_hist = px.histogram(
                            df, x=col,
                            title=f"{col}",
                            color_discrete_sequence=["#667eea"],
                            nbins=20,
                        )
                        fig_hist.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter"),
                            showlegend=False,
                            height=300,
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                # Box plots
                st.markdown("#### Box Plots")
                fig_box = px.box(
                    df[selected_cols],
                    title="Distribution & Outliers",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_box.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numeric columns detected in the dataset for charting.")

    # ── Tab 3: Correlations ──────────────────
    with tab3:
        st.subheader("🔗 Correlation Heatmap")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) >= 2:
            corr_method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman", "kendall"],
                help="Pearson for linear, Spearman for rank-based, Kendall for ordinal.",
            )
            corr_matrix = df[numeric_cols].corr(method=corr_method)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 11},
            ))
            fig_corr.update_layout(
                title=f"{corr_method.capitalize()} Correlation Matrix",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=500,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Highlight strong correlations
            st.markdown("#### 🔥 Strong Correlations (|r| > 0.7)")
            strong = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.7:
                        strong.append({
                            "Column A": corr_matrix.columns[i],
                            "Column B": corr_matrix.columns[j],
                            "Correlation": round(val, 3),
                        })
            if strong:
                st.dataframe(pd.DataFrame(strong), use_container_width=True)
            else:
                st.info("No strong correlations (|r| > 0.7) found between numeric columns.")
        else:
            st.info("Need at least 2 numeric columns to compute correlations.")

    # ── Tab 4: Data Profiling ────────────────
    with tab4:
        st.subheader("🔎 Data Profiling")

        # Column-level profiling
        profile_data = []
        for col in df.columns:
            info = {
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null": df[col].isnull().sum(),
                "Unique": df[col].nunique(),
                "Null %": f"{df[col].isnull().sum() / len(df) * 100:.1f}%",
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                info["Min"] = str(df[col].min())
                info["Max"] = str(df[col].max())
                info["Mean"] = str(round(df[col].mean(), 2))
                info["Std"] = str(round(df[col].std(), 2))
            else:
                info["Min"] = "—"
                info["Max"] = "—"
                info["Mean"] = "—"
                info["Std"] = "—"

            profile_data.append(info)

        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True)

        # Categorical column details
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.markdown("#### 📋 Categorical Column Details")
            selected_cat = st.selectbox("Select a categorical column", cat_cols)
            if selected_cat:
                value_counts = df[selected_cat].value_counts()

                col_left, col_right = st.columns(2)
                with col_left:
                    st.dataframe(
                        value_counts.reset_index().rename(
                            columns={"index": selected_cat, selected_cat: "Count"}
                        ),
                        use_container_width=True,
                    )
                with col_right:
                    fig_cat = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {selected_cat}",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig_cat.update_layout(
                        font=dict(family="Inter"),
                        height=350,
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

        # Data Filtering
        st.markdown("---")
        st.markdown("#### 🔧 Data Filter")
        filter_col = st.selectbox("Filter by column", df.columns, key="filter_col")
        if filter_col:
            unique_vals = df[filter_col].dropna().unique()
            if len(unique_vals) <= 50:
                selected_vals = st.multiselect(
                    f"Select values for '{filter_col}'",
                    options=sorted([str(v) for v in unique_vals]),
                    key="filter_vals",
                )
                if selected_vals:
                    # Cast back for numeric columns
                    if pd.api.types.is_numeric_dtype(df[filter_col]):
                        selected_vals_typed = [float(v) for v in selected_vals]
                    else:
                        selected_vals_typed = selected_vals

                    filtered_df = df[df[filter_col].isin(selected_vals_typed)]
                    st.write(f"Showing {len(filtered_df)} of {len(df)} rows")
                    st.dataframe(filtered_df, use_container_width=True)
            else:
                st.info(f"Column '{filter_col}' has {len(unique_vals)} unique values. Use text filtering below.")
                filter_text = st.text_input(f"Search in '{filter_col}'", key="filter_text")
                if filter_text:
                    mask = df[filter_col].astype(str).str.contains(filter_text, case=False, na=False)
                    filtered_df = df[mask]
                    st.write(f"Showing {len(filtered_df)} of {len(df)} rows")
                    st.dataframe(filtered_df, use_container_width=True)

    # ── Tab 5: AI-Powered Insights ───────────
    with tab5:
        st.subheader("🤖 AI-Powered Insights")

        if not api_key:
            st.warning(
                f"⚠️ Please enter your Groq API key in the sidebar to generate AI insights."
            )
        else:
            if st.button("✨ Generate AI Insights", type="primary", use_container_width=True):
                with st.spinner("Analyzing your data with AI..."):
                    try:
                        summary = build_dataset_summary(df)
                        insights = get_ai_insights(summary, api_key, model_choice)

                        # Store insights in session state for download
                        st.session_state["last_insights"] = insights

                        st.markdown("---")
                        st.markdown(insights)

                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

            # Download insights button
            if "last_insights" in st.session_state and st.session_state["last_insights"]:
                st.markdown("---")
                st.download_button(
                    label="📥 Download Insights Report",
                    data=st.session_state["last_insights"],
                    file_name="ai_insights_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # ── Tab 6: Chat with Data ────────────────
    with tab6:
        st.subheader("💬 Chat with Your Data")

        if not api_key:
            st.warning(
                f"⚠️ Please enter your {provider} API key in the sidebar to chat with your data."
            )
        else:
            # Initialize chat history in session state
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            # Build context from dataset
            dataset_summary = build_dataset_summary(df)

            # Display chat history
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message chat-ai">🤖 <strong>AI:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

            # Chat input
            user_question = st.chat_input("Ask a question about your data...")

            if user_question:
                # Add user message to history
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": user_question,
                })

                with st.spinner("Thinking..."):
                    try:
                        ai_reply = get_chat_response(
                            st.session_state["chat_history"][:-1],  # history without the latest user msg
                            user_question,
                            dataset_summary,
                            api_key,
                            model_choice,
                        )

                        # Add AI reply to history
                        st.session_state["chat_history"].append({
                            "role": "model",
                            "content": ai_reply,
                        })

                        # Rerun to display updated chat
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # Clear chat button
            if st.session_state["chat_history"]:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()

else:
    # Show a helpful message when no file is uploaded
    st.info("👆 Upload a CSV file to get started with your data analysis.")

    # Hint about the sample dataset
    st.markdown(
        "> 💡 **Tip:** A `sample_data.csv` file is included in the project folder for quick testing."
    )
