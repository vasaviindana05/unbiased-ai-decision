import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import io

# Configure Gemini
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Page config
st.set_page_config(
    page_title="Unbiased AI Decision",
    page_icon="⚖️",
    layout="wide"
)

# Google colors CSS
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.title { color: #4285F4; font-size: 3em; font-weight: bold; text-align: center; }
.subtitle { color: #34A853; font-size: 1.2em; text-align: center; }
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.bias-high { color: #EA4335; font-weight: bold; }
.bias-low { color: #34A853; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">⚖️ Unbiased AI Decision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect, Measure & Fix Hidden Bias in Your Datasets using Gemini AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://www.gstatic.com/devrel-devsite/prod/v45f61267e9184f1fcb69ee3e4a8a8a6f4e3c2d7d3c0d3c2e4a3f35a4f4c4b3c/googledevai/images/lockup.svg", width=150)
    st.title("⚙️ Settings")
    sensitive_col = st.selectbox("Select Sensitive Column", ["gender", "race", "age", "religion"])
    outcome_col = st.text_input("Outcome Column Name", "outcome")
    threshold = st.slider("Bias Threshold (80% rule)", 0.5, 1.0, 0.8)
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("🤖 Gemini AI")
    st.markdown("☁️ Google Cloud")
    st.markdown("🐍 Python + Pandas")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

with col2:
    st.subheader("📋 How It Works")
    st.markdown("""
    1. 📤 Upload CSV dataset
    2. 🔍 Select sensitive column
    3. 📊 View bias analysis
    4. 🤖 Get Gemini AI fixes
    5. 📄 Download report
    """)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head(10))

    st.markdown("---")
    st.header("📊 Bias Analysis Results")

    if sensitive_col in df.columns and outcome_col in df.columns:
        # Calculate bias
        group_rates = df.groupby(sensitive_col)[outcome_col].mean()
        majority_rate = group_rates.max()
        
        # Metrics row
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Records", df.shape[0])
        with cols[1]:
            st.metric("Groups Found", len(group_rates))
        with cols[2]:
            min_rate = group_rates.min()
            ratio = min_rate / majority_rate
            st.metric("Fairness Ratio", f"{ratio:.2f}")
        with cols[3]:
            score = int(ratio * 100)
            st.metric("Fairness Score", f"{score}/100")

        # Bias verdict
        if ratio < threshold:
            st.error(f"🚨 BIAS DETECTED! Fairness ratio {ratio:.2f} is below threshold {threshold}")
        else:
            st.success(f"✅ No significant bias detected. Fairness ratio: {ratio:.2f}")

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                x=group_rates.index,
                y=group_rates.values,
                title=f"Outcome Rate by {sensitive_col}",
                color=group_rates.values,
                color_continuous_scale=["#EA4335", "#FBBC05", "#34A853"],
                labels={"x": sensitive_col, "y": "Outcome Rate"}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(
                values=group_rates.values,
                names=group_rates.index,
                title="Distribution by Group",
                color_discrete_sequence=["#4285F4", "#EA4335", "#34A853", "#FBBC05"]
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Gemini AI Analysis
        st.markdown("---")
        st.header("🤖 Gemini AI Recommendations")
        
        if st.button("🔍 Analyze with Gemini AI", type="primary"):
            with st.spinner("Gemini AI is analyzing your dataset..."):
                prompt = f"""
                I have a dataset with bias analysis results:
                - Sensitive attribute: {sensitive_col}
                - Group outcome rates: {group_rates.to_dict()}
                - Fairness ratio: {ratio:.2f}
                - Bias detected: {ratio < threshold}
                
                Please provide:
                1. A clear explanation of the bias found
                2. Real-world impact of this bias
                3. Three specific actionable steps to fix this bias
                4. Best practices for fair AI in this context
                
                Be specific, practical and professional.
                """
                
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )
                st.markdown(response.text)

        # Download report
        st.markdown("---")
        report = f"""
UNBIASED AI DECISION - BIAS ANALYSIS REPORT
============================================
Dataset: {uploaded_file.name}
Sensitive Column: {sensitive_col}
Outcome Column: {outcome_col}

GROUP OUTCOME RATES:
{group_rates.to_string()}

FAIRNESS METRICS:
- Fairness Ratio: {ratio:.2f}
- Fairness Score: {score}/100
- Bias Detected: {ratio < threshold}
- Threshold Used: {threshold}

Generated by Unbiased AI Decision Tool
Powered by Gemini AI + Google Cloud
        """
        
        st.download_button(
            "📄 Download Bias Report",
            report,
            file_name="bias_report.txt",
            mime="text/plain"
        )
    else:
        st.warning(f"⚠️ Columns '{sensitive_col}' or '{outcome_col}' not found. Please check column names!")
        st.write("Available columns:", list(df.columns))

else:
    # Demo mode
    st.info("👆 Upload a CSV file to start bias analysis")
    st.subheader("📌 Example CSV format:")
    demo_df = pd.DataFrame({
        "gender": ["male","female","male","female","male"],
        "age": [25, 30, 35, 28, 42],
        "race": ["A","B","A","C","B"],
        "outcome": [1, 0, 1, 0, 1]
    })
    st.dataframe(demo_df)

# Footer
st.markdown("---")
st.markdown("**⚖️ Unbiased AI Decision** | Built for Google Solution Challenge 2026 | Powered by Gemini AI + Google Cloud")
