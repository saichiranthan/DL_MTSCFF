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
import plotly.express as px
import plotly.graph_objects as go

# --- Enhanced CSS for better mobile and desktop experience ---
st.markdown("""
    <style>
    /* Global styles */
    .main > div {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
        }
        .sidebar .sidebar-content {
            width: 200px !important;
        }
        .stButton>button {
            font-size: 14px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
    }
    
    /* Component styling */
    .header-container {
        background: linear-gradient(135deg, #1E90FF, #4169E1);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.5rem !important;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #1E90FF;
    }
    
    .prediction-label {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .prediction-confidence {
        font-size: 1.25rem;
        color: #666;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .action-button {
        background-color: #1E90FF;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 600;
        display: block;
        width: 100%;
        text-align: center;
    }
    
    .action-button:hover {
        background-color: #0066CC;
        transform: translateY(-2px);
    }
    
    .download-button {
        background-color: #4CAF50;
        color: white;
        text-decoration: none;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        display: inline-block;
        font-weight: 600;
        transition: all 0.3s;
        text-align: center;
        margin-top: 1rem;
    }
    
    .download-button:hover {
        background-color: #388E3C;
        transform: translateY(-2px);
    }
    
    .sensor-data-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Add tab-like styling for sections */
    .section-tabs {
        display: flex;
        margin-bottom: 1rem;
        border-bottom: 1px solid #ddd;
    }
    
    .section-tab {
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        cursor: pointer;
        border-bottom: 2px solid transparent;
    }
    
    .section-tab.active {
        border-bottom: 2px solid #1E90FF;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header with Enhanced Styling ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Department of Information Technology</h1>
        <h2 class="header-subtitle">NITK Surathkal</h2>
        <h3 style="font-weight: 400; font-size: 1.2rem; margin-bottom: 0.5rem;">Multivariate Time Series Classification with Feature Fusion</h3>
        <p style="margin: 0;">By Abhishek N B (221AI003), Sai Chiranthan HM (221AI035) </p>
    </div>
    """, unsafe_allow_html=True)

# --- Model Loading with Progress Bar ---
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        progress_bar = st.progress(0)
        model = ImprovedGIN(in_dim=6)  # Original input dimension
        progress_bar.progress(30)
        
        try:
            model.load_state_dict(torch.load("SaiChiranthan-221AI035-model.pth", 
                                          map_location="cpu", 
                                          weights_only=True))
            progress_bar.progress(100)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            progress_bar.progress(100)
            st.error(f"❌ Error loading model: {str(e)}")
            st.warning("⚠️ Using randomly initialized model")
        
        model.eval()
        return model

# --- File Processing with Error Handling ---
def process_uploaded_file(uploaded_file):
    try:
        file_content = uploaded_file.read().decode('utf-8').splitlines()
        data, _ = parse_multivariate_ts_fixed(file_content)
        data = np.array(data, dtype=np.float32)
        
        # Normalize
        scalers = [StandardScaler() for _ in range(data.shape[1])]
        for i in range(data.shape[1]):
            data[:, i, :] = scalers[i].fit_transform(data[:, i, :])
        
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# --- Sidebar with Enhanced UI ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h3 style="color: #1E90FF;">📤 Data Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload .ts file", type=["ts"], 
                                   help="Upload your multivariate time series data")
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        
        with st.expander("ℹ️ Sample Format"):
            st.code("1.0,2.0:3.0,4.0:Badminton\n2.0,3.0:4.0,5.0:Running")
    
    # Add helpful sidebar info
    st.markdown("---")
    with st.expander("📚 About This Project"):
        st.markdown("""
        This application classifies human activities using multivariate time series data from wearable sensors.
        
        **Supported Activities:**
        - Badminton
        - Running
        - Standing
        - Walking
        
        The model uses a Graph Isomorphism Network (GIN) to capture relationships between sensors.
        """)

# --- Initialize session state for tabs ---
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "immediate"

# --- Main Content with Better Organization ---
if uploaded_file:
    # Process data
    data, error = process_uploaded_file(uploaded_file)
    
    if error:
        st.error(error)
    else:
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
        
        # Make prediction once
        with torch.no_grad():
            output = model(graph_data)
            probabilities = F.softmax(output, dim=1).numpy()[0]
            pred_class_idx = torch.argmax(output).item()
            pred_class = ["Badminton", "Running", "Standing", "Walking"][pred_class_idx]
            confidence = probabilities[pred_class_idx]
        
        # Custom tab navigation
        st.markdown("""
        <div class="section-tabs">
            <div class="section-tab {}" onclick="window.location.href='?tab=immediate'">🚀 Quick Results</div>
            <div class="section-tab {}" onclick="window.location.href='?tab=full'">📊 Full Analysis</div>
        </div>
        """.format(
            "active" if st.session_state.current_tab == "immediate" else "",
            "active" if st.session_state.current_tab == "full" else ""
        ), unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Quick Results", help="Quick prediction"):
                st.session_state.current_tab = "immediate"
        
        with col2:
            if st.button("📊 Full Analysis", help="Detailed analysis"):
                st.session_state.current_tab = "full"
        
        st.markdown("---")
        
        # Immediate Results Tab
        if st.session_state.current_tab == "immediate":
            st.markdown("### 🔍 Activity Classification Results")
            
            # Create a prediction card with prominent display
            st.markdown(f"""
            <div class="prediction-card">
                <p style="margin-bottom: 0.5rem; font-size: 1rem; color: #666;">PREDICTED ACTIVITY</p>
                <div class="prediction-label">{pred_class}</div>
                <div class="prediction-confidence">Confidence: {confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add horizontal bar chart for all class probabilities
            fig = go.Figure()
            classes = ["Badminton", "Running", "Standing", "Walking"]
            colors = ["#1E90FF" if i == pred_class_idx else "#A9A9A9" for i in range(4)]
            
            fig.add_trace(go.Bar(
                y=classes,
                x=probabilities * 100,
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probabilities * 100],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Prediction Confidence by Activity",
                xaxis_title="Confidence (%)",
                margin=dict(l=0, r=0, t=40, b=0),
                height=300,
                xaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensor data visualization
            st.markdown("### 📈 Sensor Data Visualization")
            
            # Use plotly for interactive time series plot
            fig = go.Figure()
            for i in range(data.shape[1]):
                fig.add_trace(go.Scatter(
                    y=data[0, i, :50],
                    mode='lines',
                    name=f'Sensor {i+1}'
                ))
            
            fig.update_layout(
                title="Sensor Readings (First 50 Timesteps)",
                xaxis_title="Timestep",
                yaxis_title="Normalized Value",
                height=400,
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model details in collapsible section
            with st.expander("🧠 Model Architecture Details"):
                params_df, bn_df = print_model_parameters(model)
                st.write("**Model Parameters:**")
                st.dataframe(params_df, use_container_width=True)
                st.write("**Batch Normalization:**")
                st.dataframe(bn_df, use_container_width=True)
        
        # Full Analysis Tab
        elif st.session_state.current_tab == "full":
            st.markdown("### 📊 Complete Analysis Report")
            
            # Create tabs for different analysis sections
            tab1, tab2, tab3 = st.tabs(["Prediction Details", "Correlation Analysis", "Data Distribution"])
            
            with tab1:
                # Prediction details with donut chart
                st.markdown("#### Activity Classification Results")
                
                # Create two columns for layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card" style="height: 100%;">
                        <p style="margin-bottom: 0.5rem; font-size: 1rem; color: #666;">PREDICTED ACTIVITY</p>
                        <div class="prediction-label">{pred_class}</div>
                        <div class="prediction-confidence">Confidence: {confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Donut chart for probabilities
                    fig = go.Figure(data=[go.Pie(
                        labels=["Badminton", "Running", "Standing", "Walking"],
                        values=probabilities,
                        hole=.4,
                        marker_colors=['#1E90FF', '#FF7F50', '#90EE90', '#FFD700']
                    )])
                    
                    fig.update_layout(
                        title="Probability Distribution",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=250,
                        annotations=[dict(text=f"{confidence:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional prediction metrics in card format
                st.markdown("#### Prediction Metrics")
                
                # Create metrics cards in a grid
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #666;">Top Class</h4>
                        <p style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">{}</p>
                        <p style="margin: 0; color: #666;">Confidence: {:.2%}</p>
                    </div>
                    """.format(pred_class, confidence), unsafe_allow_html=True)
                
                with metric_cols[1]:
                    second_best_idx = np.argsort(probabilities)[-2]
                    second_best_class = ["Badminton", "Running", "Standing", "Walking"][second_best_idx]
                    second_best_conf = probabilities[second_best_idx]
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #666;">Second Best</h4>
                        <p style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">{}</p>
                        <p style="margin: 0; color: #666;">Confidence: {:.2%}</p>
                    </div>
                    """.format(second_best_class, second_best_conf), unsafe_allow_html=True)
            
            with tab2:
                st.markdown("#### Sensor Correlation Analysis")
                
                # Create a nicer heatmap with plotly
                fig = px.imshow(
                    kendall_corr,
                    labels=dict(x="Sensor", y="Sensor", color="Correlation"),
                    x=[f"Sensor {i}" for i in range(6)],
                    y=[f"Sensor {i}" for i in range(6)],
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                
                fig.update_layout(
                    title="Kendall Correlation Between Sensors",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Add text annotations to the heatmap
                for i in range(6):
                    for j in range(6):
                        fig.add_annotation(
                            x=i, y=j,
                            text=f"{kendall_corr[j, i]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if abs(kendall_corr[j, i]) > 0.5 else "black")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation of correlation
                with st.expander("ℹ️ Understanding Correlation Analysis"):
                    st.markdown("""
                    **Interpretation:**
                    
                    - Values close to 1 indicate strong positive correlation (sensors move together)
                    - Values close to -1 indicate strong negative correlation (sensors move in opposite directions)
                    - Values close to 0 indicate little or no correlation
                    
                    Strong correlations between sensors can indicate synchronized movement patterns characteristic of specific activities.
                    """)
            
            with tab3:
                st.markdown("#### Sensor Data Distribution")
                
                # Time series plot with all sensors
                fig = go.Figure()
                for i in range(data.shape[1]):
                    fig.add_trace(go.Scatter(
                        y=data[0, i, :],
                        mode='lines',
                        name=f'Sensor {i+1}'
                    ))
                
                fig.update_layout(
                    title="Complete Time Series Data",
                    xaxis_title="Timestep",
                    yaxis_title="Normalized Value",
                    height=400,
                    legend=dict(orientation="h", y=1.1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of sensor values
                st.markdown("#### Sensor Value Distributions")
                
                # Create histogram using plotly
                fig = go.Figure()
                for i in range(data.shape[1]):
                    fig.add_trace(go.Histogram(
                        x=data[0, i, :].flatten(),
                        name=f'Sensor {i+1}',
                        opacity=0.7,
                        nbinsx=30
                    ))
                
                fig.update_layout(
                    title="Distribution of Sensor Values",
                    xaxis_title="Normalized Value",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Generate report button
            st.markdown("### 📑 Generate Full Report")
            
            if st.button("⬇️ Generate & Download Report"):
                with st.spinner("Generating report..."):
                    # Create outputs directory
                    os.makedirs("report", exist_ok=True)
                    
                    # 1. Predictions
                    pred_df = pd.DataFrame({
                        "Class": ["Badminton", "Running", "Standing", "Walking"],
                        "Probability": probabilities
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
                        ax2.plot(data[0, i, :], label=f'Sensor {i}')
                    ax2.legend()
                    ax2.set_title("Time Series Data")
                    fig2.savefig("report/time_series_plot.png")
                    plt.close(fig2)
                    
                    # 4. Distribution plot
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    for i in range(data.shape[1]):
                        ax3.hist(data[0, i, :].flatten(), bins=30, alpha=0.5, label=f'Sensor {i}')
                    ax3.legend()
                    ax3.set_title("Data Distribution")
                    fig3.savefig("report/distribution_plot.png")
                    plt.close(fig3)
                    
                    # Create ZIP
                    zip_path = "time_series_report.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file in os.listdir("report"):
                            zipf.write(f"report/{file}", file)
                    
                    # Download button
                    with open(zip_path, "rb") as f:
                        bytes = f.read()
                        b64 = base64.b64encode(bytes).decode()
                        href = f'<div style="text-align: center; margin-top: 20px"><a href="data:application/zip;base64,{b64}" download="{zip_path}" class="download-button">⬇️ Download Full Report</a></div>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Report generated successfully!")
                    
                    # Cleanup
                    os.remove(zip_path)
else:
    # Welcome screen with instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f44b.png" width="80" style="margin-bottom: 1rem;">
        <h2>Welcome to the Time Series Classifier</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Upload a .ts file to begin your analysis</p>
        
        <div style="max-width: 500px; margin: 0 auto; text-align: left; background: #f8f9fa; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h4>Getting Started:</h4>
            <ol>
                <li>Use the file uploader in the sidebar to upload your .ts file</li>
                <li>The model will automatically process your data</li>
                <li>Choose between quick results or detailed analysis</li>
                <li>Download a full report with all insights</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)