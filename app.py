import streamlit as st
from model_code import ImprovedGIN, parse_multivariate_ts_fixed, compute_cwt, compute_kendall_correlation, print_model_parameters
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
import os
import zipfile
import base64
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# --- Mobile-Friendly CSS ---
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
        }
        .sidebar .sidebar-content {
            width: 200px !important;
        }
        .stButton>button {
            font-size: 14px;
            padding: 0.25rem;
        }
        .header-title {
            font-size: 1.5rem !important;
        }
        .header-subtitle {
            font-size: 1rem !important;
        }
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
    <div style='text-align: center; padding-bottom: 1rem;'>
        <h1 class="header-title" style='color: #1E90FF;'>Department of IT</h1>
        <h2 class="header-subtitle">NITK Surathkal</h2>
        <h3 style='font-weight: 400;'>Multivariate Time Series Classification</h3>
        <p>By Abhishek N B, Sai Chiranthan HM</p>
    </div>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    model = ImprovedGIN(in_dim=6)  # Original input dimension
    try:
        model.load_state_dict(torch.load("SaiChiranthan-221AI035-model.pth", 
                                       map_location="cpu", 
                                       weights_only=True))
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Using randomly initialized model")
    model.eval()
    return model

# --- File Processing ---
def process_uploaded_file(uploaded_file):
    file_content = uploaded_file.read().decode('utf-8').splitlines()
    data, _ = parse_multivariate_ts_fixed(file_content)
    data = np.array(data, dtype=np.float32)
    
    # Normalize
    scalers = [StandardScaler() for _ in range(data.shape[1])]
    for i in range(data.shape[1]):
        data[:, i, :] = scalers[i].fit_transform(data[:, i, :])
    
    return data

# --- Sidebar ---
with st.sidebar:
    st.header("📤 Data Input")
    uploaded_file = st.file_uploader("Upload .ts file", type=["ts"], 
                                   help="Upload your multivariate time series data")
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        with st.expander("ℹ️ Sample Format"):
            st.code("1.0,2.0:3.0,4.0:Badminton\n2.0,3.0:4.0,5.0:Running")

# --- Main Content ---
if uploaded_file:
    data = process_uploaded_file(uploaded_file)
    model = load_model()
    
    # Compute mean across time dimension (6 sensors × 100 timesteps -> 6 features)
    mean_features = np.mean(data, axis=2)  # Shape: (n_samples, 6)
    
    # Compute correlation using original data
    kendall_corr = compute_kendall_correlation(data)
    
    # Build graph (using sensor correlations)
    edge_index = torch.tensor([[i, j] for i in range(6) for j in range(6) if i != j], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([kendall_corr[i, j] for i in range(6) for j in range(6) if i != j], dtype=torch.float32)
    
    # Create graph data with mean features (6 features per sample)
    graph_data = Data(
        x=torch.tensor(mean_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Action Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Immediate Results", help="Quick prediction"):
            st.session_state.show_immediate = True
            st.session_state.show_full = False
    
    with col2:
        if st.button("📊 Full Report", help="Download complete analysis"):
            st.session_state.show_full = True
            st.session_state.show_immediate = False
    
    st.markdown("---")
    
    # Immediate Results
    if st.session_state.get('show_immediate'):
        st.markdown("### 🔍 Immediate Results")
        
        with torch.no_grad():
            output = model(graph_data)
            pred_class = ["Badminton", "Running", "Standing", "Walking"][torch.argmax(output).item()]
            confidence = torch.max(F.softmax(output, dim=1)).item()
        
        # Mobile-friendly metrics
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction</h3>
            <h2>{pred_class}</h2>
            <p>Confidence: {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sensor plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(min(3, data.shape[1])):
            ax.plot(data[0, i, :50], label=f'Sensor {i+1}')
        ax.set_title("First 50 Timesteps")
        ax.legend()
        st.pyplot(fig)
        
        # Model details expander
        with st.expander("🧠 Model Details"):
            params_df, bn_df = print_model_parameters(model)
            st.write("**Model Parameters:**")
            st.dataframe(params_df)
            st.write("**Batch Normalization:**")
            st.dataframe(bn_df)
    
    # Full Report
    if st.session_state.get('show_full'):
        st.markdown("### 📦 Generating Full Report...")
        
        with st.spinner("This may take a moment..."):
            # Create outputs directory
            os.makedirs("report", exist_ok=True)
            
            # 1. Predictions
            with torch.no_grad():
                output = model(graph_data)
                probs = F.softmax(output, dim=1).numpy()[0]
            
            pred_df = pd.DataFrame({
                "Class": ["Badminton", "Running", "Standing", "Walking"],
                "Probability": probs
            })
            pred_df.to_csv("report/predictions.csv", index=False)
            
            # 2. Correlation
            corr_df = pd.DataFrame(
                kendall_corr,
                columns=[f"Sensor {i}" for i in range(6)],
                index=[f"Sensor {i}" for i in range(6)]
            )
            corr_df.to_csv("report/correlation.csv")
            
            # 3. Plots
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
            ax1.set_title("Sensor Correlation")
            fig1.savefig("report/correlation_plot.png")
            plt.close(fig1)
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            for i in range(data.shape[1]):
                ax2.hist(data[:, i, :].flatten(), bins=30, alpha=0.5, label=f'Sensor {i}')
            ax2.legend()
            ax2.set_title("Data Distribution")
            fig2.savefig("report/distribution_plot.png")
            plt.close(fig2)
            
            # Create ZIP
            zip_path = "time_series_report.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in os.listdir("report"):
                    zipf.write(f"report/{file}", file)
            
            # Download button
            with open(zip_path, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="{zip_path}">⬇️ Download Full Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Cleanup
            os.remove(zip_path)
else:
    st.info("👋 Please upload a .ts file to begin analysis")