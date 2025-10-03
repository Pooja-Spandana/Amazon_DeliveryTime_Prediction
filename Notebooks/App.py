# Import libraries
import mlflow
import pandas as pd
import mlflow.pyfunc
import streamlit as st

# MLflow/DagsHub setup
DAGSHUB_USERNAME = "Pooja-Spandana"
DAGSHUB_REPO = "Amazon_DeliveryTime_Prediction"

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Load from run ID and artifact path
run_id = "0009aa096930489b83713cffcc379070"
logged_model = f"runs:/{run_id}/Final_RF_Model"   # the same path you used in log_model()

model = mlflow.pyfunc.load_model(logged_model)

# Feature engineering raw user data
def feature_engineering(user_input: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate all feature engineering steps used in training.
    Takes a df of raw inputs and returns a processed DataFrame.
    
    Returns: pd.DataFrame : DataFrame with engineered features included
    """
    df = user_input.copy()  

    # Create interactive features 
    df['Traffic_Area'] = df['Traffic'].astype(str) + "_" + df['Area'].astype(str)
    df['Area_Vehicle'] = df['Area'].astype(str) + "_" + df['Vehicle'].astype(str)
    df['Weather_Area'] = df['Weather'].astype(str) + "_" + df['Area'].astype(str)

    # Derived features 
    df['Is_Peak_Hours'] = df['Order_Hour'].apply(lambda x: 1 if (17 <= x <= 23) else 0)
    df['Is_Urban'] = df['Area'].apply(lambda x: 1 if x in ['Urban', 'Metropolitan'] else 0)

    return df

# Page Configuration
st.set_page_config(page_title="Amazon Delivery Prediction", page_icon="📦", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #DCDBDA; text-shadow: 2px 2px #CCCCCC; font-family: "Trebuchet MS", "Georgia";'>
        📦 Amazon Delivery Time Prediction
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("""

""")

# Sidebar
st.sidebar.title("📘 About This App")

st.sidebar.markdown("""
This application uses a **RandomForest Regression Model** 
to predict Amazon delivery times based on various factors like: **Agent details**, **Order details** & **Environment** 
### 📊 Model Info
- Algorithm: **RandomForest (HP Tuned)**  
- Performance:  
  - RMSE ≈ **22 hrs**  
  - R² ≈ **0.82** (explains ~82% of variance)  
- Generalizes well (no overfitting). 
---
### ⚙️ How to Use:
1. Fill in the agent, order & environment details in the form.  
2. Click **Predict** to get an estimated delivery time (in hours).  
3. Use the prediction to plan delivery schedules & resource allocation. 
""")

# Button Configuration
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #176A9C;
        color: #DCDBDA;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Label color
st.markdown(
    """
    <style>
    label {color: white !important; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True
)

# Page Layout (3 fixed boxes)
col1, col2 = st.columns([2, 1])   # Form on left, prediction/suggestions on right

with col1:
    with st.form("delivery_form"):
        st.markdown("""
                    <h3 style='text-align: center; color: #DCDBDA; font-family: "Trebuchet MS", "Georgia";'> 
                        Enter Delivery Details 
                    </h3>
                    """, unsafe_allow_html=True)
        
        # Nested columns inside col1
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            agent_age = st.slider("Agent Age", 15, 60, 30, label_visibility="visible")
            order_hour = st.slider("Order Hour", 0, 23, 14, label_visibility="visible")

        with subcol2:
            agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1, label_visibility="visible")
            distance = st.slider("Distance (km)", 10.0, 7000.0, 10.0, label_visibility="visible")
        
        subcol3, subcol4 = st.columns(2)
        with subcol3:
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Stormy", "Windy", "Sandstorms"], label_visibility="visible")
            traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"], label_visibility="visible")

        with subcol4:
            vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "van"], label_visibility="visible")
            area = st.selectbox("Area", ["Urban", "Semi-Urban", "Other", "Metropolitan"], label_visibility="visible")

        category = st.selectbox("Product Category", [
            "Electronics", "Books", "Jewelry", "Toys", "Snacks", "Skincare",
            "Outdoors", "Apparel", "Sports", "Grocery", "Pet Supplies", "Home",
            "Cosmetics", "Kitchen", "Clothing", "Shoes"
        ], label_visibility="visible")
        
        submitted = st.form_submit_button("PREDICT")

with col2:
    # Prediction Box
    st.markdown(
    """
    <div style='background-color:#2e3135f2; padding:10px; border-radius:10px; font-family: "Trebuchet MS", "Georgia"; color:#DCDBDA;'>
        <h3 style='text-align:center; color:#DCDBDA;'>Prediction</h3>
    </div>
    """,
    unsafe_allow_html=True
    )
    
    if submitted:
        input_data = pd.DataFrame([{
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Distance_km": distance,
            "Order_Hour": order_hour,
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category
        }])

        input_df = feature_engineering(input_data)
        pred = model.predict(input_df)[0]
        
        # Convert to days + hours format
        days = int(pred // 24)
        hours = int(pred % 24)

        # Set background & text color based on thresholds
        if 0 < days <= 5:
            bg_color = "#155724"   
            text_color = "#FBFDFC" 
            suggestions = [
                "✅ Delivery looks on time.",
                "✅ No major risks expected.",
                "📦 Continue monitoring agent performance for consistency."
            ]
        elif 5 < days <= 8:
            bg_color = "#B48B12"   
            text_color = "#0e0405"
            suggestions = [
                "⚠️ Delivery slightly delayed.",
                "🚦 Avoid peak-hour orders for better efficiency.",
                "📍 Recheck routes & traffic conditions."
            ] 
        else:
            bg_color = "#721c24"   
            text_color = "#0e0405" 
            suggestions = [
                "❌ Delivery highly delayed!",
                "🚚 Consider reassigning to a different agent/vehicle.",
                "🌦️ Check traffic/weather conditions immediately.",
                "🛠️ Operational changes may be required."
            ]

        # Prediction Box
        st.markdown(
            f"""
            <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; font-size:18px; color:{text_color};'>
                🚚 <b>Estimated Delivery Time:</b><br>
                {pred:.2f} hours (~{days} days {hours} hours)
            </div>
            """,
            unsafe_allow_html=True
        )

        # Suggestions Box
        st.markdown(
            f"""
            <div style='background-color:#2e3135f2; padding:20px; border-radius:10px; font-family:"Trebuchet MS","Georgia"; color:#DCDBDA;'>
                <h4 style='text-align:center; color:#DCDBDA;'>Suggestions</h4>
                <ul style='text-align:left;'>
                    {''.join([f"<li>{s}</li>" for s in suggestions])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("Prediction will appear here once you submit the form.")
