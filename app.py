import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from PIL import Image
import base64

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="BigMart Sales Prediction System",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Main Container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Prediction Box */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .prediction-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Info Box */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model_paths = ["bigmart_best_model.pkl", "model.pkl", "bigmart_model.pkl"]
        model = None
        version = None
        
        for path in model_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                        model, version = loaded_data
                    else:
                        model = loaded_data
                        version = "Unknown"
                break
        
        if model is None:
            st.error("❌ Model file not found! Please upload the model file.")
            return None, None
        
        return model, version
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def validate_inputs(item_weight, item_visibility, item_mrp):
    """Validate user inputs"""
    errors = []
    
    if item_weight < 1 or item_weight > 25:
        errors.append("Item Weight should be between 1kg and 25kg")
    
    if item_visibility < 0 or item_visibility > 0.3:
        errors.append("Item Visibility should be between 0 and 0.3")
    
    if item_mrp < 30 or item_mrp > 300:
        errors.append("Item MRP should be between $30 and $300")
    
    return errors

def create_visualizations(input_data, prediction):
    """Create visualizations for the prediction"""
    
    # Feature importance visualization
    fig_radar = go.Figure()
    
    categories = ['Item Weight', 'Item Visibility', 'Item MRP', 'Outlet Age']
    values = [
        input_data['Item_Weight'].iloc[0] / 25 * 100,
        input_data['Item_Visibility'].iloc[0] / 0.3 * 100,
        input_data['Item_MRP'].iloc[0] / 300 * 100,
        input_data['Outlet_Age'].iloc[0] / 40 * 100
    ]
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Input Values',
        line_color='#667eea'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Feature Impact Analysis",
        height=400
    )
    
    # Sales range visualization
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        title={'text': "Predicted Sales ($)"},
        delta={'reference': 2000},
        gauge={
            'axis': {'range': [None, 5000]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 1500], 'color': "#ffcccc"},
                {'range': [1500, 3000], 'color': "#ffffcc"},
                {'range': [3000, 5000], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3500
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    
    return fig_radar, fig_gauge

# ==================== DATA LOADING ====================
model, model_version = load_model()

if model is None:
    with st.expander("📤 Upload Model File"):
        uploaded_file = st.file_uploader("Upload your trained model file (.pkl)", type=['pkl'])
        if uploaded_file is not None:
            try:
                model = pickle.load(uploaded_file)
                st.success("✅ Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {e}")
    st.stop()

# ==================== MAIN APPLICATION ====================
# Header
st.markdown("""
    <div class="main-header">
        <h1>🏪 BigMart Sales Prediction System</h1>
        <p>Advanced Machine Learning Solution for Retail Sales Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942786.png", width=80)
    st.markdown("## 🎯 Prediction Parameters")
    st.markdown("---")
    
    # Mode Selection
    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["🎨 Interactive Form", "📊 Batch Prediction", "📈 Historical Analysis"],
        index=0
    )
    
    st.markdown("---")
    
    if prediction_mode == "🎨 Interactive Form":
        st.markdown("### 📝 Input Features")
        
        # Item Features Section
        with st.expander("🛒 Item Features", expanded=True):
            item_weight = st.slider(
                "Item Weight (kg)", 
                min_value=1.0, 
                max_value=25.0, 
                value=12.5, 
                step=0.5,
                help="Weight of the product in kilograms"
            )
            
            item_fat_content = st.radio(
                "Item Fat Content", 
                ('Low Fat', 'Regular'),
                horizontal=True
            )
            
            item_visibility = st.slider(
                "Item Visibility (%)", 
                min_value=0.0, 
                max_value=30.0, 
                value=7.0, 
                step=1.0,
                format="%.1f%%"
            ) / 100.0
            
            item_type = st.selectbox(
                "Item Category", 
                [
                    'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 
                    'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
                    'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
                    'Breads', 'Starchy Foods', 'Others', 'Seafood'
                ]
            )
            
            item_mrp = st.number_input(
                "Item MRP ($)", 
                min_value=30.0, 
                max_value=300.0, 
                value=150.0, 
                step=5.0,
                help="Maximum Retail Price in dollars"
            )
        
        # Outlet Features Section
        with st.expander("🏪 Outlet Features", expanded=True):
            current_year = datetime.now().year
            outlet_establishment_year = st.select_slider(
                "Outlet Establishment Year", 
                options=list(range(1985, current_year+1)),
                value=2010
            )
            
            outlet_size = st.selectbox(
                "Outlet Size", 
                ('Small', 'Medium', 'High'),
                help="Physical size category of the outlet"
            )
            
            outlet_location_type = st.selectbox(
                "Location Type", 
                ('Tier 1', 'Tier 2', 'Tier 3'),
                help="Tier category of the city"
            )
            
            outlet_type = st.selectbox(
                "Outlet Type", 
                ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store']
            )
        
        predict_button = st.button("🔮 Predict Sales", use_container_width=True)
        
        if predict_button:
            st.session_state['predict_clicked'] = True
            st.session_state['input_data'] = {
                'item_weight': item_weight,
                'item_fat_content': item_fat_content,
                'item_visibility': item_visibility,
                'item_type': item_type,
                'item_mrp': item_mrp,
                'outlet_establishment_year': outlet_establishment_year,
                'outlet_size': outlet_size,
                'outlet_location_type': outlet_location_type,
                'outlet_type': outlet_type
            }
    
    elif prediction_mode == "📊 Batch Prediction":
        st.markdown("### 📁 Batch Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file with input data",
            type=['csv'],
            help="File should contain all required features"
        )
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            
            if st.button("🚀 Run Batch Prediction"):
                st.session_state['batch_clicked'] = True
                st.session_state['batch_data'] = batch_data
    
    elif prediction_mode == "📈 Historical Analysis":
        st.markdown("### 📊 Historical Data Analysis")
        st.info("Upload historical sales data for analysis and insights")
        
        hist_file = st.file_uploader("Upload historical sales data", type=['csv'])
        if hist_file is not None:
            hist_data = pd.read_csv(hist_file)
            st.session_state['hist_data'] = hist_data

# Main Content Area
if prediction_mode == "🎨 Interactive Form":
    if 'predict_clicked' in st.session_state and st.session_state['predict_clicked']:
        input_dict = st.session_state['input_data']
        
        # Validate inputs
        errors = validate_inputs(
            input_dict['item_weight'],
            input_dict['item_visibility'],
            input_dict['item_mrp']
        )
        
        if errors:
            for error in errors:
                st.error(f"⚠️ {error}")
        else:
            # Prepare input data
            input_data = pd.DataFrame([{
                'Item_Identifier': 'FDX07',
                'Item_Weight': input_dict['item_weight'],
                'Item_Fat_Content': input_dict['item_fat_content'],
                'Item_Visibility': input_dict['item_visibility'],
                'Item_Type': input_dict['item_type'],
                'Item_MRP': input_dict['item_mrp'],
                'Outlet_Identifier': 'OUT027',
                'Outlet_Establishment_Year': input_dict['outlet_establishment_year'],
                'Outlet_Size': input_dict['outlet_size'],
                'Outlet_Location_Type': input_dict['outlet_location_type'],
                'Outlet_Type': input_dict['outlet_type'],
            }])
            
            # Feature Engineering
            input_data['Outlet_Age'] = datetime.now().year - input_data['Outlet_Establishment_Year']
            input_data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Create two columns for results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display prediction
                    st.markdown("""
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted Sales</div>
                            <div class="prediction-value">${:,.2f}</div>
                        </div>
                    """.format(prediction), unsafe_allow_html=True)
                    
                    # Sales category
                    if prediction < 1000:
                        st.markdown("""
                            <div class="warning-box">
                                📉 <strong>Low Sales Prediction</strong><br>
                                Consider promotional strategies or price optimization.
                            </div>
                        """, unsafe_allow_html=True)
                    elif prediction < 3000:
                        st.markdown("""
                            <div class="info-box">
                                📈 <strong>Moderate Sales Prediction</strong><br>
                                Good potential. Monitor performance regularly.
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="success-box">
                                🚀 <strong>High Sales Prediction!</strong><br>
                                Excellent potential! Ensure adequate stock availability.
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Key metrics
                    st.markdown("### 📊 Key Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Outlet Age", f"{input_data['Outlet_Age'].iloc[0]} years")
                    with metric_col2:
                        st.metric("Item MRP", f"${input_dict['item_mrp']:.2f}")
                    with metric_col3:
                        visibility_percent = input_dict['item_visibility'] * 100
                        st.metric("Visibility", f"{visibility_percent:.1f}%")
                
                with col2:
                    # Create visualizations
                    fig_radar, fig_gauge = create_visualizations(input_data, prediction)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Additional insights
                st.markdown("---")
                st.markdown("### 🔍 Detailed Analysis")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col4:
                    st.markdown("### 💡 Business Recommendations")
                    recommendations = []
                    
                    if input_dict['item_visibility'] < 0.05:
                        recommendations.append("• Increase product visibility on shelves")
                    if input_dict['item_mrp'] > 200:
                        recommendations.append("• Consider price optimization for better sales")
                    if input_data['Outlet_Age'].iloc[0] < 5:
                        recommendations.append("• New outlet - focus on customer acquisition")
                    if input_dict['item_weight'] > 15:
                        recommendations.append("• Heavy item - consider bundling with lighter items")
                    
                    for rec in recommendations:
                        st.write(rec)
                    
                    if not recommendations:
                        st.write("✓ Current configuration looks optimal!")
                
                # Export option
                st.markdown("---")
                if st.button("📥 Export Prediction Report"):
                    report_data = {
                        'Prediction': prediction,
                        'Input_Parameters': input_dict,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="Download Report (CSV)",
                        data=csv,
                        file_name=f"sales_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Reset button
                if st.button("🔄 New Prediction"):
                    del st.session_state['predict_clicked']
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("""
            <div class="card">
                <h2>🎯 Welcome to BigMart Sales Prediction System</h2>
                <p>This advanced system uses machine learning to predict item sales across different outlets.</p>
                
                <h3>✨ Features:</h3>
                <ul>
                    <li>Real-time sales prediction based on product and outlet characteristics</li>
                    <li>Interactive visualizations and analytics</li>
                    <li>Business recommendations based on predictions</li>
                    <li>Export reports for further analysis</li>
                </ul>
                
                <h3>🚀 How to use:</h3>
                <ol>
                    <li>Fill in the product details in the sidebar</li>
                    <li>Provide outlet characteristics</li>
                    <li>Click "Predict Sales" to get your prediction</li>
                    <li>Review detailed analytics and recommendations</li>
                </ol>
                
                <div class="info-box">
                    💡 <strong>Pro Tip:</strong> Try different combinations to see how changes in MRP 
                    or visibility affect predicted sales!
                </div>
            </div>
        """, unsafe_allow_html=True)

elif prediction_mode == "📊 Batch Prediction":
    if 'batch_clicked' in st.session_state and st.session_state['batch_clicked']:
        batch_data = st.session_state['batch_data']
        
        with st.spinner("Processing batch predictions..."):
            # Process batch data
            predictions = []
            for idx, row in batch_data.iterrows():
                try:
                    input_data = pd.DataFrame([row])
                    if 'Outlet_Establishment_Year' in input_data.columns:
                        input_data['Outlet_Age'] = datetime.now().year - input_data['Outlet_Establishment_Year']
                        input_data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
                    
                    pred = model.predict(input_data)[0]
                    predictions.append(pred)
                except:
                    predictions.append(None)
            
            batch_data['Predicted_Sales'] = predictions
            
            # Display results
            st.markdown("### 📊 Batch Prediction Results")
            st.dataframe(batch_data, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            valid_preds = [p for p in predictions if p is not None]
            if valid_preds:
                with col1:
                    st.metric("Total Items", len(valid_preds))
                with col2:
                    st.metric("Average Sales", f"${np.mean(valid_preds):.2f}")
                with col3:
                    st.metric("Max Sales", f"${np.max(valid_preds):.2f}")
                with col4:
                    st.metric("Min Sales", f"${np.min(valid_preds):.2f}")
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        if st.button("🔄 New Batch"):
            del st.session_state['batch_clicked']
            st.rerun()

elif prediction_mode == "📈 Historical Analysis":
    if 'hist_data' in st.session_state:
        hist_data = st.session_state['hist_data']
        
        st.markdown("### 📊 Historical Data Insights")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(hist_data))
        with col2:
            if 'Sales' in hist_data.columns or 'Item_Outlet_Sales' in hist_data.columns:
                sales_col = 'Sales' if 'Sales' in hist_data.columns else 'Item_Outlet_Sales'
                st.metric("Average Sales", f"${hist_data[sales_col].mean():.2f}")
        with col3:
            if 'Item_MRP' in hist_data.columns:
                st.metric("Average MRP", f"${hist_data['Item_MRP'].mean():.2f}")
        
        # Distribution plots
        if 'Item_MRP' in hist_data.columns:
            fig_mrp = px.histogram(hist_data, x='Item_MRP', title='Item MRP Distribution', 
                                  color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig_mrp, use_container_width=True)
        
        if 'Outlet_Type' in hist_data.columns:
            outlet_counts = hist_data['Outlet_Type'].value_counts()
            fig_outlet = px.bar(x=outlet_counts.index, y=outlet_counts.values, 
                               title='Sales by Outlet Type',
                               color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig_outlet, use_container_width=True)
        
        if st.button("Clear Data"):
            del st.session_state['hist_data']
            st.rerun()
    else:
        st.info("👈 Please upload historical sales data from the sidebar to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p>© 2024 BigMart Sales Prediction System | Powered by Machine Learning | Built with Streamlit</p>
        <p style='font-size: 0.8rem; opacity: 0.7;'>Model Version: {} | Last Updated: {}</p>
    </div>
""".format(model_version if model_version else "Unknown", datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
