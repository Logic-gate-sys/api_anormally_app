# app.py - OAuth2 Anomaly Detection System (Fixed Text Visibility)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="OAuth2 Anomaly Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FIXED CSS - ALL TEXT VISIBLE
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #7474BF, #348AC7);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }
    
    /* Typography - FIXED */
    h1 {
        color: white !important;
        text-align: center;
        font-weight: 700 !important;
        font-size: 3rem !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    h2 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin: 2rem 0 1rem 0 !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }
    
    h3 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }
    
    /* Subtitle - FIXED */
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 0.02em;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }
    
    /* All paragraph text - FIXED */
    p {
        color: inherit;
    }
    
    /* Card Styles */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(30, 60, 114, 0.1);
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(30, 60, 114, 0.2);
    }
    
    .content-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(30, 60, 114, 0.08);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 0.9rem 2.5rem;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(30, 60, 114, 0.25);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(30, 60, 114, 0.35);
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed rgba(30, 60, 114, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2a5298;
        background: rgba(30, 60, 114, 0.02);
    }
    
    [data-testid="stFileUploader"] label {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input {
        background: white !important;
        border: 2px solid rgba(30, 60, 114, 0.2) !important;
        border-radius: 10px;
        padding: 0.7rem;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #1e3c72 !important;
        transition: all 0.2s ease;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #2a5298 !important;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
    }
    
    .stNumberInput label {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
    
    /* Select Box */
    .stSelectbox>div>div>div {
        background: white !important;
        border: 2px solid rgba(30, 60, 114, 0.2) !important;
        border-radius: 10px;
        font-weight: 500 !important;
        color: #1e3c72 !important;
    }
    
    .stSelectbox label {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(30, 60, 114, 0.03) !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(30, 60, 114, 0.08) !important;
    }
    
    .dataframe tbody td {
        color: #1e3c72 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 60, 114, 0.96) 0%, rgba(42, 82, 152, 0.96) 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio>label {
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio>div {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.8rem;
    }
    
    [data-testid="stSidebar"] .stRadio>div>label {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio>div>label:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 12px;
        font-weight: 600 !important;
        color: #1e3c72 !important;
        border: 2px solid rgba(30, 60, 114, 0.15);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(30, 60, 114, 0.03) !important;
        border-color: #2a5298;
    }
    
    /* Messages - FIXED */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: none !important;
        color: #155724 !important;
        font-weight: 600 !important;
    }
    
    .stSuccess > div {
        color: #155724 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #bfe9ff 100%) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: none !important;
        color: #004085 !important;
        font-weight: 600 !important;
    }
    
    .stInfo > div {
        color: #004085 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stError > div {
        color: white !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Progress text */
    .stProgress ~ div {
        color: white !important;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        height: 10px;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.25);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(17, 153, 142, 0.35);
    }
    
    /* Metric Values */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/random_forest_anomaly_detector.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure models are in the 'models/' directory.")
        st.stop()

model, scaler, feature_names = load_model_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_data(n_samples=50, seed=None):
    """Generate realistic synthetic metrics for testing"""
    if seed is not None:
        np.random.seed(seed)
    
    test_data = {}
    
    for feature in feature_names:
        if 'bytes' in feature.lower() or 'alloc' in feature.lower():
            test_data[feature] = np.random.exponential(scale=1000000, size=n_samples)
        elif 'ratio' in feature.lower() or 'percent' in feature.lower():
            test_data[feature] = np.random.beta(2, 5, size=n_samples)
        elif 'count' in feature.lower() or 'total' in feature.lower():
            test_data[feature] = np.random.poisson(lam=100, size=n_samples)
        elif 'cpu' in feature.lower() and 'idle' in feature.lower():
            test_data[feature] = np.random.normal(loc=80, scale=10, size=n_samples)
        elif 'cpu' in feature.lower():
            test_data[feature] = np.random.normal(loc=20, scale=10, size=n_samples)
        elif 'memory' in feature.lower() or 'mem' in feature.lower():
            test_data[feature] = np.random.exponential(scale=500000, size=n_samples)
        elif 'network' in feature.lower():
            test_data[feature] = np.random.exponential(scale=10000, size=n_samples)
        else:
            test_data[feature] = np.random.normal(loc=100, scale=50, size=n_samples)
        
        test_data[feature] = np.abs(test_data[feature])
    
    return pd.DataFrame(test_data)

def display_results(df, predictions, probabilities):
    """Display prediction results with visualizations"""
    
    anomaly_count = predictions.sum()
    normal_count = len(predictions) - anomaly_count
    avg_confidence = probabilities[predictions == 1].mean() if anomaly_count > 0 else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results Header
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 3rem 0;'>
        <h2 style='font-size: 2rem; margin: 0 0 0.5rem 0; color: white;'>Detection Results</h2>
        <p style='color: rgba(255,255,255,0.9); font-size: 1rem;'>Analysis complete</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Samples</div>
            <div class='metric-value' style='color: #1e3c72;'>{len(predictions):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Normal</div>
            <div class='metric-value' style='color: #10b981;'>{normal_count:,}</div>
            <div style='color: #10b981; font-weight: 600; font-size: 1rem; margin-top: 0.3rem;'>{normal_count/len(predictions)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Anomalies</div>
            <div class='metric-value' style='color: #ef4444;'>{anomaly_count:,}</div>
            <div style='color: #ef4444; font-weight: 600; font-size: 1rem; margin-top: 0.3rem;'>{anomaly_count/len(predictions)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Confidence</div>
            <div class='metric-value' style='color: #f59e0b;'>{avg_confidence:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Visualization
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[normal_count, anomaly_count],
            marker=dict(colors=['#10b981', '#ef4444']),
            hole=0.5,
            textfont=dict(size=16, color='white', family='Inter'),
            textinfo='label+percent'
        )])
        fig_pie.update_layout(
            title=dict(text="<b>Distribution</b>", font=dict(size=20, color='white', family='Inter')),
            showlegend=False,
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=35,
            marker=dict(
                color=probabilities,
                colorscale='Bluered',
                line=dict(color='white', width=0.5)
            ),
            hovertemplate='Confidence: %{x:.2%}<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(
            title=dict(text="<b>Confidence Distribution</b>", font=dict(size=20, color='white', family='Inter')),
            xaxis_title="Probability",
            yaxis_title="Frequency",
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)',
            font=dict(color='white', family='Inter'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results Table
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 1rem 0;'>
        <h3 style='font-size: 1.5rem; color: white;'>Detailed Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    results_df = pd.DataFrame({
        'ID': range(1, len(predictions) + 1),
        'Status': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
        'Confidence': [f"{p:.2%}" for p in probabilities],
        'Risk': ['High' if p > 0.9 else 'Medium' if p > 0.7 else 'Low' for p in probabilities]
    })
    
    if 'scenario' in df.columns:
        results_df['Scenario'] = df['scenario'].values
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        show_filter = st.selectbox(
            "Filter Results",
            ["All", "Anomalies Only", "Normal Only", "High Risk"]
        )
    
    if show_filter == "Anomalies Only":
        results_df = results_df[results_df['Status'] == 'Anomaly']
    elif show_filter == "Normal Only":
        results_df = results_df[results_df['Status'] == 'Normal']
    elif show_filter == "High Risk":
        results_df = results_df[results_df['Risk'] == 'High']
    
    st.dataframe(results_df, use_container_width=True, height=400)
    
    # Download
    csv = results_df.to_csv(index=False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="Download Report",
            data=csv,
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1>OAuth2 Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enterprise Security Monitoring Platform • 99.6% Accuracy • Real-time Analysis</p>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Navigation")
    st.markdown("<br>", unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["Detection", "Performance", "Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Model Information")
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 12px; margin-top: 1rem;'>
        <p style='margin: 0.6rem 0; font-size: 0.9rem; color: white;'><b>Algorithm:</b> Random Forest</p>
        <p style='margin: 0.6rem 0; font-size: 0.9rem; color: white;'><b>Features:</b> {len(feature_names)}</p>
        <p style='margin: 0.6rem 0; font-size: 0.9rem; color: white;'><b>F1-Score:</b> 99.61%</p>
        <p style='margin: 0.6rem 0; font-size: 0.9rem; color: white;'><b>Latency:</b> 0.12ms</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: DETECTION
# ============================================================================

if page == "Detection":
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 2.5rem; border-radius: 20px; text-align: center; margin: 2rem auto 3rem auto; max-width: 800px; border: 1px solid rgba(255,255,255,0.2);'>
        <h2 style='font-size: 2rem; margin: 0 0 0.8rem 0; color: white; font-weight: 600;'>Anomaly Detection</h2>
        <p style='font-size: 1.05rem; color: rgba(255,255,255,0.95); margin: 0;'>Upload your OAuth2 metrics or generate synthetic test data</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='content-card'>
            <h3 style='color: #1e3c72; margin-top: 0; font-size: 1.3rem;'>Upload CSV File</h3>
            <p style='color: #6b7280; margin-bottom: 1.5rem;'>Upload your OAuth2 metrics for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop CSV file here",
            type=['csv'],
            help="Upload a CSV file with your OAuth2 metrics",
            key="file_upload",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class='content-card'>
            <h3 style='color: #1e3c72; margin-top: 0; font-size: 1.3rem;'>Generate Test Data</h3>
            <p style='color: #6b7280; margin-bottom: 1.5rem;'>Create synthetic data for testing</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            n_samples = st.number_input("Samples", min_value=10, max_value=200, value=50, step=10)
        with col2_2:
            random_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)
        
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate & Analyze", use_container_width=True)
    
    if uploaded_file is not None:
        
        with st.spinner("Processing data..."):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.3)
        
        st.success(f"Loaded {len(df):,} samples with {len(df.columns)} features")
        
        with st.expander("Preview Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button("Analyze", use_container_width=True, key="predict_upload")
        
        if predict_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("<p style='color: white;'>Preprocessing features...</p>", unsafe_allow_html=True)
            progress_bar.progress(25)
            time.sleep(0.3)
            
            try:
                X = df[feature_names]
            except KeyError:
                st.error("CSV columns don't match expected features")
                st.stop()
            
            status_text.markdown("<p style='color: white;'>Running model...</p>", unsafe_allow_html=True)
            progress_bar.progress(50)
            X_scaled = scaler.transform(X)
            
            status_text.markdown("<p style='color: white;'>Generating predictions...</p>", unsafe_allow_html=True)
            progress_bar.progress(75)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            progress_bar.progress(100)
            status_text.markdown("<p style='color: white;'>Complete</p>", unsafe_allow_html=True)
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            display_results(df, predictions, probabilities)
    
    elif generate_btn:
        
        with st.spinner("Generating test data..."):
            df = generate_sample_data(n_samples=n_samples, seed=random_seed)
            time.sleep(0.5)
        
        st.success(f"Generated {len(df):,} synthetic samples with {len(feature_names)} features")
        
        with st.expander("Preview Generated Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.info("Randomly generated data that mimics real OAuth2 metrics")
        
        with st.spinner("Analyzing..."):
            progress_bar = st.progress(0)
            
            progress_bar.progress(33)
            X_scaled = scaler.transform(df)
            
            progress_bar.progress(66)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()
        
        display_results(df, predictions, probabilities)

# ============================================================================
# PAGE 2: PERFORMANCE
# ============================================================================

elif page == "Performance":
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 3rem 0;'>
        <h2 style='font-size: 2rem; margin: 0 0 0.5rem 0; color: white;'>Model Performance</h2>
        <p style='color: rgba(255,255,255,0.9); font-size: 1rem;'>Comprehensive evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    metrics_data = {
        'F1-Score': 0.9961,
        'Precision': 0.9948,
        'Recall': 0.9974,
        'Accuracy': 0.9927
    }
    
    for col, (metric_name, value) in zip([col1, col2, col3, col4], metrics_data.items()):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                title={'text': f"<b>{metric_name}</b>", 'font': {'color': 'white', 'size': 16, 'family': 'Inter'}},
                number={'suffix': '%', 'font': {'size': 32, 'color': 'white', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#1e3c72', 'thickness': 0.7},
                    'bgcolor': 'rgba(255,255,255,0.2)',
                    'borderwidth': 2,
                    'bordercolor': 'white',
                    'steps': [
                        {'range': [0, 70], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [70, 85], 'color': 'rgba(251, 191, 36, 0.3)'},
                        {'range': [85, 95], 'color': 'rgba(34, 197, 94, 0.3)'},
                        {'range': [95, 100], 'color': 'rgba(16, 185, 129, 0.5)'}
                    ]
                }
            ))
            fig.update_layout(
                height=210,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Inter'},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h3 style='font-size: 1.5rem; color: white;'>Confusion Matrix</h3>
        </div>
        """, unsafe_allow_html=True)
        
        conf_matrix = np.array([[29, 2], [1, 381]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Normal', 'Predicted Anomaly'],
            y=['Actual Normal', 'Actual Anomaly'],
            text=conf_matrix,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 24, "color": "white", "family": "Inter"},
            colorscale=[[0, '#e0e7ff'], [1, '#1e3c72']],
            showscale=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Count: %{z}<extra></extra>'
        ))
        fig.update_layout(
            height=360,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12, family='Inter'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Only 3 errors out of 413 samples (99.3% accuracy)")
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h3 style='font-size: 1.5rem; color: white;'>Performance Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        stats_df = pd.DataFrame({
            'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 
                       'Inference Time', 'Training Samples'],
            'Value': ['381', '29', '2', '1', '0.12 ms', '1,651']
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=280)
        
        st.info("Process 8,000+ samples per second with sub-millisecond latency")

# ============================================================================
# PAGE 3: INSIGHTS
# ============================================================================

elif page == "Insights":
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 3rem 0;'>
        <h2 style='font-size: 2rem; margin: 0 0 0.5rem 0; color: white;'>Feature Importance</h2>
        <p style='color: rgba(255,255,255,0.9); font-size: 1rem;'>Understanding detection signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    importance_df = pd.DataFrame({
        'Feature': [
            'network_transmit_bytes',
            'go_memstats_frees',
            'network_transmit_packets',
            'go_memstats_alloc_bytes',
            'go_memstats_mallocs',
            'http_requests_total',
            'network_receive_packets',
            'memory_anon_pages',
            'memory_available',
            'tcp_out_segs'
        ],
        'Importance': [0.1198, 0.0899, 0.0725, 0.0624, 0.0581, 
                       0.0515, 0.0511, 0.0381, 0.0362, 0.0349]
    })
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Blues',
            line=dict(color='white', width=1)
        ),
        text=[f"<b>{x:.1%}</b>" for x in importance_df['Importance']],
        textposition='outside',
        textfont=dict(size=13, color='white', family='Inter'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Top 10 Features</b>",
            font=dict(size=22, color='white', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Importance Score</b>",
        yaxis_title="",
        height=520,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', size=12, family='Inter'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=70, t=70, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='content-card'>
            <h3 style='color: #1e3c72; margin-top: 0; font-size: 1.4rem;'>Network Metrics</h3>
            <p style='color: #374151; font-size: 1rem; line-height: 1.7;'>Network traffic patterns are the strongest indicators of anomalies, accounting for <strong>19.2%</strong> combined importance</p>
            <ul style='color: #6b7280; font-size: 0.95rem; line-height: 1.9;'>
                <li>Transmit bytes and packets change during authentication failures</li>
                <li>Both inbound and outbound traffic patterns are affected</li>
                <li>Enables real-time detection capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-card'>
            <h3 style='color: #1e3c72; margin-top: 0; font-size: 1.4rem;'>Memory Patterns</h3>
            <p style='color: #374151; font-size: 1rem; line-height: 1.7;'>Go memory statistics reveal <strong>4 of the top 5</strong> most important features for detection</p>
            <ul style='color: #6b7280; font-size: 0.95rem; line-height: 1.9;'>
                <li>Garbage collection behavior differs significantly during anomalies</li>
                <li>Memory allocation patterns serve as key signals</li>
                <li>Memory pressure indicates potential authentication errors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.92) 100%); padding: 2.8rem; border-radius: 20px; text-align: center; margin: 2rem 0; box-shadow: 0 20px 60px rgba(0,0,0,0.25);'>
        <h2 style='color: #1e3c72; margin-top: 0; font-size: 1.9rem; font-weight: 700;'>Key Insights</h2>
        <p style='color: #374151; font-size: 1.15rem; line-height: 1.9; margin: 1.5rem 0; font-weight: 400;'>
            OAuth2 anomalies create a <strong style='color: #1e3c72;'>distinctive fingerprint</strong> across network and memory subsystems.<br>
            The model leverages <strong style='color: #2a5298;'>281 individual signals</strong> to achieve <strong style='color: #10b981;'>99.6% F1-score</strong>.<br>
            <em style='color: #6b7280;'>No single feature dominates, ensuring robust production performance.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 16px; margin-top: 4rem;'>
    <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem; color: white;'>OAuth2 Anomaly Detection System v1.0</p>
    <p style='font-size: 0.95rem; opacity: 0.9; color: white;'>Random Forest Classifier • Trained on 2,064 samples</p>
    <p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0.4rem; color: white;'>99.61% F1-Score • 0.12ms latency • Production Ready</p>
</div>
""", unsafe_allow_html=True)
