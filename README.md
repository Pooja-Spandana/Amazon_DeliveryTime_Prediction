# <div align="center">***Amazon Delivery Time Prediction***</div>
---

## ***Problem Statement***

An end-to-end **Machine Learning project** that predicts the **estimated delivery time (in hours)** for Amazon deliveries based on multiple factors like distance, weather, traffic, and delivery agent details.  

The project demonstrates a complete **data science workflow** including data cleaning, feature engineering, model training, experiment tracking with **MLflow (via DagsHub)**, and deployment using **Streamlit**.  

---

## 🚀 Project Overview  
- **Objective**: Predict delivery times to improve logistics efficiency and provide accurate ETAs.  
- **Data**: Historical delivery records containing agent details, order info, traffic, weather, and delivery outcomes.  
- **Approach**:
  1. Data Cleaning & EDA 📊  
  2. Feature Engineering ✨ (interactive + derived features)  
  3. Model Training ⚡ (Random Forest, XGBoost, etc.)  
  4. Experiment Tracking 🔍 (MLflow + DagsHub)  
  5. Deployment 🌐 (Streamlit app for real-time predictions)  

---

## 🛠️ Tech Stack  
- **Languages/Tools**: Python (3.10), Jupyter, Streamlit  
- **Libraries**:  
  - Data: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Modeling: `scikit-learn`, `xgboost`  
  - Tracking: `mlflow`, `dagshub`  
  - Deployment: `streamlit`  

---

## ***Business Use Cases***

- ***Enhanced Delivery Logistics:*** Predict delivery times to improve customer satisfaction and optimize delivery schedules.
- ***Dynamic Traffic and Weather Adjustments:*** Adjust delivery estimates based on current traffic and weather conditions.
- ***Agent Performance Evaluation:*** Evaluate agent efficiency and identify areas for training or improvement.
- ***Operational Efficiency:*** Optimize resource allocation for deliveries by analyzing trends and performance metrics.

---
## ***Approach***

### `Data_cleaning_&_EDA.ipynb`

  - ***Data Preparation:***
      - Load and preprocess the dataset.
      - Handle missing or inconsistent data.
      - Derive meaningful features from raw data(e.g., calculating distance between store and drop locations).
  - ***Data Cleaning:***
      - Remove duplicates and handle missing values.
      - Handle placeholders in categorical variables (e.g., 'Nan', 'null').
  - ***Exploratory Data Analysis (EDA):***
      - Analyze trends in delivery times, agent performance, and external factors.
      - Visualize the impact of traffic, weather, and other variables on delivery times.
      - Use statistical tests to understand relationship between features.

### `Model_building_&_training.ipynb`

  - ***Feature Engineering:***
      - Create new interactive features based on the insights gained during EDA.
      - Extract time-based features (e.g., hour of day, day of the week).
  - ***Train-Val-Test split***
      - Train - 70%
      - Val - 15 %
      - Test - 15%
  - ***Build & Train multiple regression models, including:***
      - Linear Regression
      - Ridge
      - Random Forest Regressor
      - Gradient Boosting Regressor
      - XGBoost Regressor
  - ***Evaluate models using metrics:***
      - RMSE 
      - MAE
      - R-squared.
  - ***Model Comparison and Tracking:***
      - Use MLflow to log, compare, and manage different regression models.
      - Log the hyperparameters, performance metrics, features used and the model.
  - ***Feaature Importance & Hyperparameter Tuning***
      - Find feature importance for the best performing models and create a feature set common for both
      - HP tune the top 2 performing models (Random Forest & XGB) on the new feataure set
      - Compare results and pick the best model to retain on Train+Val data.
      - Finally evaluate on Test data (only once) and plot residual plots for better inferencing results
### `App.py`

  - ***Application Development & Deployment:***
      - Built a user interface using Streamlit to get real-time delivery time predictions.
        - ***Features:***
          - User inputs delivery details (sliders + dropdowns)
          - Predicts delivery time in hours + days format
          - Color-coded result box:
            - 🟢 Green → On-time (1–5 days)
            - 🟠 Orange → Slightly delayed (6–8 days)
            - 🔴 Red → High delay risk (>9 days)
          - Helpful suggestions box (e.g., avoid peak hours, vans for long distances)
      - Deployed the application in streamlit for accessibility and scalability.
--- 
## ***📊 Results & Insights***

### ***Final Model: Random Forest Regressor (with Hyperparameter Tuning) - Achieved strong and consistent performance across training and testing datasets.***

### ***✅ Key Results***
- **Train Performance:**
  - RMSE ≈ **20.97 hrs**
  - MAE ≈ **16.21 hrs**
  - R² ≈ **0.84**
- **Test Performance:**
  - RMSE ≈ **21.99 hrs**
  - MAE ≈ **16.98 hrs**
  - R² ≈ **0.82**
### ***🔎 Interpretation***
- **Consistency (Generalization):**
  - Train RMSE ≈ 20.9 vs Test RMSE ≈ 21.9 → very small gap.  
  - Train R² ≈ 0.83 vs Test R² ≈ 0.82 → nearly identical.  
  - ✅ Model is **not overfitting** and generalizes well to unseen data.
- **Error Magnitude:**
  - RMSE ≈ 22 hrs → predictions deviate by ~22 hours on average.  
  - MAE ≈ 17 hrs → median error is lower and robust to outliers.  
  - Considering deliveries vary widely (short vs long trips), this is a reasonable error margin.
- **Residual Analysis:**
  - Residuals are centered around 0 → unbiased predictions.  
  - Some heteroscedasticity observed (variance increases for longer trips), which is expected in real-world logistics data.  
  - Handles **short/medium deliveries well**, while variance grows for long-distance orders.
### ***🏆 Conclusion***
- The **HP-tuned Random Forest model** performs robustly and generalizes effectively.  
- Explains **~82% of variance in delivery times**.  
- Achieves a **good trade-off** between bias and variance.  
- Errors remain **stable across the test set**, confirming reliability for production deployment.  
---
## ***🙌 Acknowledgements***

- **_Raw Dataset:_** https://drive.google.com/file/d/1W-iJDAoFJRfT9vGELLk08xsGT_bOkogt/view
  
- ***Cleaned Dataset:*** https://github.com/Pooja-Spandana/Amazon_DeliveryTime_Prediction/blob/main/Data/Processed/EDA_dataset.csv
  
- ***Tracked with Mlflow DagsHub:*** https://dagshub.com/Pooja-Spandana/Amazon_DeliveryTime_Prediction.mlflow/#/experiments/0?viewStateShareKey=86aea5967d61069f6fa9d5a86db6c6cf5c7ac433dcfc7b5a88c354d9e4287853&compareRunsMode=TABLE
  
- ***Deployed via Streamlit Cloud:*** https://ps-amazondeliverytime-prediction.streamlit.app/
---
