import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# === Page Configuration ===
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for better styling ===
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #1E88E5;
    }
    .prediction-value {
        font-size: 3rem;
        color: #00FF00;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# === 1. Load the trained model with caching ===
@st.cache_resource
def load_model():
    """Load the pre-trained model with caching for better performance."""
    try:
        with open("bigmart_best_model.pkl", "rb") as f:
            model, version = pickle.load(f)
        return model, version
    except FileNotFoundError:
        st.error("❌ Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        return None, None

# Load model
model, version = load_model()

if model is None:
    st.stop()

# Display model info in sidebar
st.sidebar.success(f"✅ Model loaded successfully")
st.sidebar.info(f"📦 scikit-learn version: {version}")

# === 2. App Header ===
st.markdown('<h1 class="main-header">🏪 BigMart Sales Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Predict the sales of an item at a specific outlet using machine learning.
        Fill in the details below and click 'Predict Sales' to get your prediction.
    </div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# === 3. Input Form ===
with col1:
    st.markdown("### 📝 Input Features")
    st.markdown("Enter the details of the item and the outlet:")
    
    with st.form(key="prediction_form"):
        # Item Features
        st.markdown("#### 🛒 Item Features")
        item_weight = st.number_input(
            "Item Weight (kg)", 
            min_value=1.0, 
            max_value=25.0, 
            value=15.5, 
            step=0.1,
            help="Weight of the item in kilograms"
        )
        
        item_fat_content = st.selectbox(
            "Item Fat Content", 
            ('Low Fat', 'Regular'),
            help="Fat content level of the item"
        )
        
        item_visibility = st.number_input(
            "Item Visibility", 
            min_value=0.0, 
            max_value=0.3, 
            value=0.07, 
            step=0.01,
            format="%.3f",
            help="Percentage of total display area occupied by the product (0-0.3)"
        )
        
        item_type = st.selectbox(
            "Item Type", 
            [
                'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
                'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
                'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
                'Starchy Foods', 'Others', 'Seafood'
            ],
            help="Category of the item"
        )
        
        item_mrp = st.number_input(
            "Item MRP ($)", 
            min_value=30.0, 
            max_value=300.0, 
            value=150.0, 
            step=1.0,
            help="Maximum Retail Price in dollars"
        )
        
        # Outlet Features
        st.markdown("#### 🏪 Outlet Features")
        current_year = datetime.now().year
        outlet_establishment_year = st.number_input(
            "Outlet Establishment Year", 
            min_value=1985, 
            max_value=current_year, 
            value=2010, 
            step=1,
            help="Year when the outlet was established"
        )
        
        outlet_size = st.selectbox(
            "Outlet Size", 
            ('Small', 'Medium', 'High'),
            help="Physical size of the outlet"
        )
        
        outlet_location_type = st.selectbox(
            "Outlet Location Type", 
            ('Tier 1', 'Tier 2', 'Tier 3'),
            help="Tier category of the city where outlet is located"
        )
        
        outlet_type = st.selectbox(
            "Outlet Type", 
            [
                'Supermarket Type1', 'Supermarket Type2', 
                'Supermarket Type3', 'Grocery Store'
            ],
            help="Type of retail outlet"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)

# === 4. Prediction Logic ===
with col2:
    if submitted:
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame([{
            'Item_Identifier': 'FDX07',  # Placeholder for one-hot encoding
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': 'OUT027',  # Placeholder for one-hot encoding
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type,
        }])
        
        # Feature Engineering
        input_data['Outlet_Age'] = current_year - input_data['Outlet_Establishment_Year']
        input_data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
        
        # Input validation
        st.markdown("### 📊 Input Summary")
        
        # Display input summary in a nice format
        summary_df = pd.DataFrame({
            'Feature': ['Item Weight', 'Item Fat Content', 'Item Visibility', 'Item Type', 
                       'Item MRP', 'Outlet Age', 'Outlet Size', 'Outlet Location', 'Outlet Type'],
            'Value': [f"{item_weight} kg", item_fat_content, f"{item_visibility:.3f}", 
                     item_type, f"${item_mrp:.2f}", f"{current_year - outlet_establishment_year} years",
                     outlet_size, outlet_location_type, outlet_type]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Make prediction
        try:
            with st.spinner("Calculating prediction..."):
                prediction = model.predict(input_data)[0]
            
            # Display prediction result
            st.markdown("### 🎯 Predicted Item Outlet Sales")
            
            # Create a nice prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Estimated Sales</div>
                <div class="prediction-value">${prediction:,.2f}</div>
                <div class="info-text">This prediction is based on the input features provided above.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add interpretation
            if prediction < 1000:
                st.info("📉 This item is predicted to have lower sales. Consider promotional strategies.")
            elif prediction < 3000:
                st.success("📈 This item is predicted to have moderate sales.")
            else:
                st.success("🚀 This item is predicted to have high sales potential!")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
            st.info("Please check if all input values are valid and try again.")

# === 5. Additional Information in Expanders ===
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **BigMart Sales Prediction Model**
    
    This app uses a pre-trained machine learning model to predict the sales of items in BigMart outlets.
    
    **Features used in prediction:**
    - Item properties (weight, fat content, visibility, type, MRP)
    - Outlet characteristics (size, location type, type, age)
    
    **How to use:**
    1. Fill in all the input features in the left panel
    2. Click 'Predict Sales' button
    3. View the predicted sales amount and interpretation
    
    **Note:** Predictions are estimates based on historical data and may vary from actual sales.
    """)

with st.expander("📖 Feature Guide"):
    st.markdown("""
    **Feature Descriptions:**
    
    - **Item Weight**: Weight of the product in kilograms
    - **Item Fat Content**: Whether the item is low fat or regular
    - **Item Visibility**: Percentage of display area occupied (0-0.3)
    - **Item Type**: Category of the food/beverage item
    - **Item MRP**: Maximum Retail Price in dollars
    - **Outlet Age**: How many years the outlet has been operating
    - **Outlet Size**: Small, Medium, or High based on store area
    - **Outlet Location Type**: Tier 1 (metro), Tier 2 (cities), or Tier 3 (towns)
    - **Outlet Type**: Supermarket Type1/2/3 or Grocery Store
    """)

# === 6. Footer ===
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with ❤️ using Streamlit | Machine Learning Model | BigMart Sales Prediction
    </div>
""", unsafe_allow_html=True)
