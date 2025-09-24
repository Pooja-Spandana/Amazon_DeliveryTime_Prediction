# <div align="center">***Amazon Delivery Time Prediction***</div>
---

## ***Problem Statement***

This project aims to predict delivery times for e-commerce orders based on a variety of factors such as product size, distance, traffic conditions, and shipping method. Using the provided dataset, let's preprocess, analyze, and build regression models to accurately estimate delivery times. The final application will allow users to input relevant details and receive estimated delivery times via a user-friendly interface.

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
      - Perform feature engineering (e.g., calculating distance between store and drop locations).
  - ***Data Cleaning:***
      - Remove duplicates and handle missing values.
      - Standardize categorical variables (e.g., weather, traffic).
  - ***Exploratory Data Analysis (EDA):***
      - Analyze trends in delivery times, agent performance, and external factors.
      - Visualize the impact of traffic, weather, and other variables on delivery times.
  - ***Feature Engineering:***
      - Calculate geospatial distances using store and drop coordinates.
      - Extract time-based features (e.g., hour of day, day of the week).

### `Model_building_&_training.ipynb`

  - ***Build & Train multiple regression models, including:***
      - Linear Regression
      - Random Forest Regressor
      - Gradient Boosting Regressor
  - ***Evaluate models using metrics:***
      - RMSE 
      - MAE
      - R-squared.
  - ***Model Comparison and Tracking:***
      - Use MLflow to log, compare, and manage different regression models.
      - Document the hyperparameters, performance metrics, and model versions.

### `Streamlit.py`

  - ***Application Development & Deployment:***
      - Built a user interface using Streamlit to input order details (e.g., distance, traffic, weather, etc.) and display predicted delivery times.
      - Deployed the application in streamlit for accessibility and scalability.
--- 