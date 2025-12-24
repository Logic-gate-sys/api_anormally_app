# app.py - OAuth2 Anomaly Detection Streamlit App (Beautiful UI)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="OAuth2 Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODERN CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background - Animated Gradient */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1 {
        color: white !important;
        text-align: center;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    h2 {
        color: white !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 3rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Card Styles - Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* White Cards */
    .white-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .white-card:hover {
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #666 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* Buttons - Modern Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        border: 3px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Number Input */
    .stNumberInput>div>div>input {
        background: white;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        transition: all 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Select Box */
    .stSelectbox>div>div>div {
        background: white;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(102, 126, 234, 0.05) !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio>label {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio>div>label:hover {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 15px;
        font-weight: 600;
        color: #667eea !important;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.05);
        border-color: #667eea;
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 15px;
        padding: 1rem;
        border: none;
        color: #155724 !important;
        font-weight: 600;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 1rem;
        border: none;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-radius: 15px;
        padding: 1rem;
        border: none;
        color: white !important;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 12px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #667eea;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
    }
    
    /* Spinner */
    .stSpinner>div {
        border-top-color: white !important;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(17, 153, 142, 0.5);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown {
        animation: fadeIn 0.6s ease-out;
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
        st.error("⚠️ Model files not found. Please ensure models are saved in 'models/' directory.")
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Header with icon
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='font-size: 2.5rem; margin: 0;'>📊 Detection Results</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Analysis complete - here's what we found</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics cards with custom styling
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class='white-card' style='text-align: center;'>
            <p style='color: #999; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;'>Total Samples</p>
            <p style='font-size: 3rem; font-weight: 800; color: #667eea; margin: 0;'>{len(predictions):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='white-card' style='text-align: center;'>
            <p style='color: #999; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;'>🟢 Normal</p>
            <p style='font-size: 3rem; font-weight: 800; color: #10b981; margin: 0;'>{normal_count:,}</p>
            <p style='color: #10b981; font-weight: 600; font-size: 1.1rem;'>{normal_count/len(predictions)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='white-card' style='text-align: center;'>
            <p style='color: #999; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;'>🔴 Anomalies</p>
            <p style='font-size: 3rem; font-weight: 800; color: #ef4444; margin: 0;'>{anomaly_count:,}</p>
            <p style='color: #ef4444; font-weight: 600; font-size: 1.1rem;'>{anomaly_count/len(predictions)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence = probabilities[predictions == 1].mean() if anomaly_count > 0 else 0
        st.markdown(f"""
        <div class='white-card' style='text-align: center;'>
            <p style='color: #999; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;'>Avg Confidence</p>
            <p style='font-size: 3rem; font-weight: 800; color: #f59e0b; margin: 0;'>{avg_confidence:.0%}</p>
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
            textfont=dict(size=18, color='white', family='Inter'),
            textinfo='label+percent'
        )])
        fig_pie.update_layout(
            title=dict(
                text="<b>Detection Distribution</b>",
                font=dict(size=22, color='white', family='Inter')
            ),
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=40,
            marker=dict(
                color=probabilities,
                colorscale='Turbo',
                line=dict(color='white', width=0.5)
            ),
            hovertemplate='<b>Confidence: %{x:.2%}</b><br>Count: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(
            title=dict(
                text="<b>Confidence Distribution</b>",
                font=dict(size=22, color='white', family='Inter')
            ),
            xaxis_title="Anomaly Probability",
            yaxis_title="Frequency",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)',
            font=dict(color='white', family='Inter'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results table with modern styling
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 1rem 0;'>
        <h3 style='font-size: 1.8rem;'>📋 Detailed Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    results_df = pd.DataFrame({
        'ID': range(1, len(predictions) + 1),
        'Status': ['🔴 Anomaly' if p == 1 else '🟢 Normal' for p in predictions],
        'Confidence': [f"{p:.2%}" for p in probabilities],
        'Risk': ['🔥 High' if p > 0.9 else '⚠️ Medium' if p > 0.7 else '✅ Low' for p in probabilities]
    })
    
    if 'scenario' in df.columns:
        results_df['Scenario'] = df['scenario'].values
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        show_filter = st.selectbox(
            "🔍 Filter Results",
            ["All", "Anomalies Only", "Normal Only", "High Risk"],
            label_visibility="visible"
        )
    
    if show_filter == "Anomalies Only":
        results_df = results_df[results_df['Status'] == '🔴 Anomaly']
    elif show_filter == "Normal Only":
        results_df = results_df[results_df['Status'] == '🟢 Normal']
    elif show_filter == "High Risk":
        results_df = results_df[results_df['Risk'] == '🔥 High']
    
    st.dataframe(results_df, use_container_width=True, height=400)
    
    # Download button
    csv = results_df.to_csv(index=False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Complete Report",
            data=csv,
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1>🛡️ OAuth2 Anomaly Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Security Monitoring • 99.6% Accuracy • Real-time Detection</p>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Navigation")
    st.markdown("<br>", unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🎯 Detect Anomalies", "📊 Performance Dashboard", "🔍 Feature Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📌 Quick Stats")
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
        <p style='margin: 0.5rem 0;'><b>Model:</b> Random Forest</p>
        <p style='margin: 0.5rem 0;'><b>Features:</b> {len(feature_names)}</p>
        <p style='margin: 0.5rem 0;'><b>F1-Score:</b> 99.61%</p>
        <p style='margin: 0.5rem 0;'><b>Speed:</b> 0.12ms/sample</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: DETECTION
# ============================================================================

if page == "🎯 Detect Anomalies":
    
    # Modern hero section
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 3rem; border-radius: 25px; text-align: center; margin: 2rem auto 3rem auto; max-width: 800px; border: 1px solid rgba(255,255,255,0.2);'>
        <h2 style='font-size: 2.2rem; margin: 0 0 1rem 0; color: white;'>Ready to Detect Threats?</h2>
        <p style='font-size: 1.2rem; color: rgba(255,255,255,0.9); margin: 0;'>Upload your metrics or generate test data to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two options with better layout
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='white-card'>
            <h3 style='color: #667eea; margin-top: 0; font-size: 1.5rem;'>📤 Upload CSV File</h3>
            <p style='color: #666; margin-bottom: 1.5rem;'>Upload your OAuth2 metrics for real-time analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type=['csv'],
            help="Upload a CSV file with your OAuth2 metrics",
            key="file_upload",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class='white-card'>
            <h3 style='color: #764ba2; margin-top: 0; font-size: 1.5rem;'>🎲 Generate Test Data</h3>
            <p style='color: #666; margin-bottom: 1.5rem;'>Try the model with synthetic data instantly</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            n_samples = st.number_input("Samples", min_value=10, max_value=200, value=50, step=10)
        with col2_2:
            random_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)
        
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🎲 Generate & Analyze", use_container_width=True)
    
    # Handle file upload
    if uploaded_file is not None:
        
        with st.spinner("📥 Processing your data..."):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.3)
        
        st.success(f"✅ Successfully loaded **{len(df):,}** samples with **{len(df.columns)}** features")
        
        with st.expander("👀 Preview Your Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button("🔮 Analyze for Anomalies", use_container_width=True, key="predict_upload")
        
        if predict_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Preprocessing features...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            try:
                X = df[feature_names]
            except KeyError:
                st.error("❌ CSV columns don't match expected features")
                st.stop()
            
            status_text.text("🧠 Running ML model...")
            progress_bar.progress(50)
            X_scaled = scaler.transform(X)
            
            status_text.text("🎯 Generating predictions...")
            progress_bar.progress(75)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            display_results(df, predictions, probabilities)
    
    # Handle generate test data
    elif generate_btn:
        
        with st.spinner("🎲 Generating synthetic test data..."):
            df = generate_sample_data(n_samples=n_samples, seed=random_seed)
            time.sleep(0.5)
        
        st.success(f"✅ Generated **{len(df):,}** synthetic samples with **{len(feature_names)}** features")
        
        with st.expander("👀 Preview Generated Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.info("💡 This is randomly generated data that mimics real OAuth2 metrics")
        
        with st.spinner("🔮 Analyzing data..."):
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
# PAGE 2: DASHBOARD
# ============================================================================

elif page == "📊 Performance Dashboard":
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='font-size: 2.5rem; margin: 0;'>📊 Model Performance</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Comprehensive metrics and evaluation results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics with gauges
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    metrics_data = {
        'F1-Score': (0.9961, '99.61%'),
        'Precision': (0.9948, '99.48%'),
        'Recall': (0.9974, '99.74%'),
        'Accuracy': (0.9927, '99.27%')
    }
    
    for col, (metric_name, (value, display)) in zip([col1, col2, col3, col4], metrics_data.items()):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                title={'text': f"<b>{metric_name}</b>", 'font': {'color': 'white', 'size': 18, 'family': 'Inter'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': 'white', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#667eea', 'thickness': 0.75},
                    'bgcolor': 'rgba(255,255,255,0.2)',
                    'borderwidth': 3,
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
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Inter'},
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Confusion Matrix and Stats
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h3 style='font-size: 1.8rem;'>🎯 Confusion Matrix</h3>
        </div>
        """, unsafe_allow_html=True)
        
        conf_matrix = np.array([[29, 2], [1, 381]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Normal', 'Predicted Anomaly'],
            y=['Actual Normal', 'Actual Anomaly'],
            text=conf_matrix,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 28, "color": "white", "family": "Inter"},
            colorscale=[[0, '#e0e7ff'], [1, '#667eea']],
            showscale=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Count: %{z}<extra></extra>'
        ))
        fig.update_layout(
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13, family='Inter'),
            xaxis=dict(side='bottom'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Only **3 errors** out of 413 samples — **99.3% accuracy**!")
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h3 style='font-size: 1.8rem;'>⚡ Performance Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        stats_df = pd.DataFrame({
            'Metric': ['✅ True Positives', '✅ True Negatives', '❌ False Positives', '❌ False Negatives', 
                       '⚡ Inference Time', '📊 Training Samples'],
            'Value': ['381', '29', '2', '1', '0.12 ms', '1,651']
        })
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        st.info("💡 **Real-time capable**: Process **8,000+ samples per second**")
        st.info("🚀 **Production ready**: Sub-millisecond latency per prediction")

# ============================================================================
# PAGE 3: INSIGHTS
# ============================================================================

elif page == "🔍 Feature Insights":
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='font-size: 2.5rem; margin: 0;'>🔍 Feature Importance</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'>Understand what drives anomaly detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance data
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
    
    # Modern bar chart
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Plasma',
            line=dict(color='white', width=1.5),
            cornerradius=5
        ),
        text=[f"<b>{x:.1%}</b>" for x in importance_df['Importance']],
        textposition='outside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Top 10 Most Important Features</b>",
            font=dict(size=24, color='white', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Importance Score</b>",
        yaxis_title="",
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', size=13, family='Inter'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=20, r=80, t=80, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Insights cards with better design
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='white-card'>
            <h3 style='color: #667eea; margin-top: 0; font-size: 1.6rem;'>🌐 Network Metrics</h3>
            <p style='color: #444; font-size: 1.05rem; line-height: 1.8;'>Network traffic is the <strong>#1 indicator</strong> of anomalies with <strong>19.2%</strong> combined importance</p>
            <ul style='color: #666; font-size: 1rem; line-height: 2;'>
                <li>📡 Transmit bytes/packets change during auth failures</li>
                <li>🔄 Both inbound and outbound traffic affected</li>
                <li>⚡ Real-time detection is possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='white-card'>
            <h3 style='color: #764ba2; margin-top: 0; font-size: 1.6rem;'>💾 Memory Patterns</h3>
            <p style='color: #444; font-size: 1.05rem; line-height: 1.8;'>Go memory stats reveal <strong>4 of top 5 features</strong> indicating critical signals</p>
            <ul style='color: #666; font-size: 1rem; line-height: 2;'>
                <li>🗑️ GC behavior differs significantly in anomalies</li>
                <li>📦 Allocation patterns are key signals</li>
                <li>⚠️ Memory pressure indicates potential errors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Key takeaway with stunning design
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); padding: 3rem; border-radius: 25px; text-align: center; margin: 2rem 0; box-shadow: 0 20px 60px rgba(0,0,0,0.3); border: 2px solid rgba(255,255,255,0.3);'>
        <h2 style='color: #667eea; margin-top: 0; font-size: 2.2rem; font-weight: 800;'>🎯 Key Takeaway</h2>
        <p style='color: #333; font-size: 1.3rem; line-height: 2; margin: 1.5rem 0; font-weight: 500;'>
            OAuth2 anomalies create a <strong style='color: #667eea;'>distinctive fingerprint</strong> across network and memory subsystems.<br>
            The model combines <strong style='color: #764ba2;'>281 weak signals</strong> into a robust detector with <strong style='color: #10b981;'>99.6% F1-score</strong>.<br>
            <em style='color: #666;'>No single feature dominates, ensuring production resilience and reliability.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; margin-top: 4rem;'>
    <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>🛡️ OAuth2 Anomaly Detection System v1.0</p>
    <p style='font-size: 1rem; opacity: 0.9;'>Built with Streamlit & Random Forest • Trained on 2,064 samples</p>
    <p style='font-size: 0.95rem; opacity: 0.8; margin-top: 0.5rem;'>99.61% F1-Score • 0.12ms inference • Production Ready</p>
</div>
""", unsafe_allow_html=True)
