
# RideFare AI – Predictive Analytics & BI for Smarter Taxi Pricing

---

<img width="1233" height="340" alt="RideFare AI" src="https://github.com/user-attachments/assets/cc51c4a1-6527-41bc-812b-9210feb3181a" />


1. ## Business context

Ride-hailing and taxi platforms set (or validate) fares based on distance, traffic, time of day, and operational rules. Inaccurate fare estimates create real business risk:

* **Customer trust & conversion:** Underestimates trigger bill shock; overestimates reduce booking conversion.
* **Supply balance:** Poor price signals distort driver incentives and reduce platform liquidity.
* **Fraud & leakage:** Implausible fares (e.g., very long trips with very low prices) slip through without analytics controls.
* **Support costs:** Disputes and manual reviews rise when pricing is inconsistent.

This project demonstrates how **data analytics + machine learning** can produce **reliable, explainable fare predictions** and arm business users with **Power BI dashboards** to monitor demand patterns, identify anomalies, and track model performance—all geared to improving pricing accuracy and customer experience.

## Problem statement

> Build a predictive system that estimates the **final fare** for a taxi/ride-hailing trip using pickup/drop-off coordinates, trip distance, timestamp features, and passenger count; surface insights and performance diagnostics through interactive **Power BI** dashboards.

1.1 ## Who benefits (stakeholders)

* **Pricing/Revenue** teams: tighter, data-driven fare curves and guardrails.
* **Operations**: visibility into peak hours, high-error segments, and driver/passenger dynamics.
* **Risk & Compliance**: anomaly detection for suspicious trips and outlier fares.
* **Support/CX**: fewer fare disputes; faster resolution with evidence.
* **Leadership**: KPI view of accuracy (MAE/RMSE/MAPE) and business impact.

1.2 ## Business objectives & success criteria

* **Accuracy:** Reduce average fare error (MAE/RMSE) vs. a simple baseline (e.g., linear fare rule).
* **Consistency:** Lower error variance across **hours**, **distance buckets**, and **passenger counts**.
* **Explainability:** Provide feature importance and error diagnostics (e.g., which hours or trip types are hardest).
* **Actionability:** Deliver **Power BI** pages for (1) executive KPIs, (2) demand & fare insights, (3) model diagnostics.

1.3 **Key KPIs**

* **MAE (₹/trip):** average absolute deviation from actual fare.
* **RMSE (₹/trip):** error with stronger penalty for large mistakes.
* **MAPE (%):** scale-free percentage error for comparability across fare levels.
* **Coverage:** share of trips with error within ±10% / ±₹X.
* **Anomaly rate:** proportion of implausible fare–distance combinations caught.

1.4 ## Scope & assumptions

* **Data inputs:** pickup/drop-off coordinates, pickup datetime, passenger count; engineered features (distance, hour, weekday/weekend, distance buckets).
* **Models:** baseline linear model → tree-based models (Random Forest, XGBoost).
* **BI layer:** Power BI dashboards for EDA, business KPIs, and model diagnostics.

1.5 **Assumptions**

* Historical data is representative of production patterns.
* Fare rules are reasonably stable during the modeling window.
* No surge pricing field is provided; time-of-day/week proxies capture demand effects.

1.6 **Out of scope (for this version)**

* Real-time traffic, weather, and surge factors (can be added later).
* Routing optimization and ETA prediction.

1.7 ## Business impact narrative

Accurate fare prediction tightens pricing controls and reduces leakage. BI visibility enables teams to **spot high-error segments** (e.g., very short or very long trips, late-night hours) and take action—whether by retraining, feature enrichment (traffic/weather), or policy tweaks. Ultimately, the platform sees **higher booking conversion, fewer disputes, and improved driver & rider satisfaction**.

---

# 2. Objective

The project is designed with a dual purpose:

### 2.1. Predictive Modeling

Develop a machine learning model capable of estimating **taxi fares** using key trip attributes such as:

* Pickup and drop-off latitude/longitude
* Trip distance (engineered feature)
* Pickup time and derived temporal features (hour, weekday/weekend, month)
* Passenger count

The model compares baseline approaches (Linear Regression) with more advanced ensemble algorithms (Random Forest, XGBoost) to identify which best balances accuracy and interpretability.

### 2.2. Analytical & Business Intelligence Layer

Beyond building a predictive model, the project integrates **data visualization** and **business analytics** through **Power BI dashboards**, enabling stakeholders to:

* Explore demand and fare patterns across time, distance, and passenger groups
* Identify anomalies (e.g., unusually low fares for long trips)
* Monitor model performance across different segments
* Track KPIs such as MAE, RMSE, and error distribution to ensure consistency

### 2.3. End Goals

* **Accuracy:** Deliver reliable fare estimates that reduce errors compared to naïve distance-only rules.
* **Business Insight:** Provide actionable visibility into operational patterns (peak hours, passenger trends, fare anomalies).
* **Scalability:** Establish a framework that can be extended with external features (weather, traffic, surge pricing) and integrated into production.

---

## 3. Workflow

This project was carried out in a structured, end-to-end data science pipeline. Each stage was designed to ensure that the model not only achieves high predictive accuracy but also provides business value through interpretable insights and visualizations.

### **3.1. Data Cleaning & Exploratory Data Analysis (EDA)**

* Inspected the raw dataset of \~1M trip records for missing values, duplicates, and invalid entries.
* Handled incomplete GPS coordinates (dropoff missing values) and corrected outliers such as negative fares, zero distances, or trips extending far outside New York City boundaries.
* Performed **exploratory analysis** to understand distributions of fares, trip distances, passenger counts, and temporal usage patterns.
* Visualized key trends to spot anomalies and detect seasonal/temporal patterns in taxi operations.

**Outcome:** A clean dataset with realistic trips, ready for feature engineering and modeling.

### **3.2. Feature Engineering**

* **Distance Metrics:** Calculated trip distance using the **Haversine formula** (GPS-based great-circle distance). Added a log-transformed version (`log_distance`) to address skewness.
* **Time-Based Features:** Extracted hour, day of week, and month from pickup timestamps. Created binary flags for **weekend trips** and **nighttime trips**, as fare dynamics differ during these periods.
* **Interaction Features:** Designed `distance × passenger_count` to capture the effect of group travel on fare patterns.
* **Geospatial Features (optional extension):** Pickup and dropoff coordinates could be clustered into “zones” to represent popular areas such as airports or downtown.

**Outcome:** A richer feature set capturing spatial, temporal, and behavioral patterns that influence fare amounts.

### **3.3. Machine Learning Modeling**

* **Baseline Model:** Started with Linear Regression to establish a benchmark for performance.
* **Tree-Based Models:**

  * **Random Forest Regressor** – leveraged for non-linear relationships and interpretability through feature importance.
  * **XGBoost Regressor** – tuned with cross-validation for optimized performance, especially on complex patterns in the data.

**Outcome:** Built robust models capable of handling the high variability in real-world taxi trip fares.

### **3.4. Model Evaluation**

* Used multiple metrics for a holistic evaluation:

  * **RMSE (Root Mean Squared Error):** To capture the magnitude of large errors.
  * **MAE (Mean Absolute Error):** To report the average prediction error in interpretable dollar values.
  * **MAPE (Mean Absolute Percentage Error):** To express model error relative to fare size, useful for comparing across trip types.
* Conducted **residual analysis** and error distribution checks by hour, distance, and passenger count to detect where the models struggled (e.g., very short trips, late-night rides).

**Outcome:** Quantified model accuracy, identified strengths and weaknesses, and compared Random Forest vs. XGBoost performance.

### **3.5. Visualization & Business Intelligence (Power BI)**

* Built an **interactive Power BI dashboard** to bridge technical outputs and business insights.
* Key dashboards included:

  * **Fare trends** by distance buckets, time of day, and passenger count.
  * **Model performance monitoring** – actual vs predicted fares, error distribution, and comparative accuracy of models.
  * **Operational KPIs** – busiest hours, most common trip distances, and revenue-driving trip segments.
* The dashboard allows non-technical stakeholders to quickly interpret results and apply them in decision-making.

**Outcome:** Delivered a complete analytics solution combining predictive modeling with business-oriented visualization.

---

### 4. Key Results (Business + Technical)

* **Exploratory Data Analysis (EDA):**
  The dataset contained over **1 million trips** with variables such as fare, geolocations, datetime, and passenger counts.

  * Median fare was **\$8.5**, with a long tail reaching up to **\$500**.
  * Most trips were **short-distance, single-passenger rides**, while fares tended to rise moderately with higher passenger counts (6 passengers averaged the highest fares at \~\$12.25).

* **Model Performance:**
  Three models were benchmarked for fare prediction:

  * **Linear Regression:** RMSE 5.64
  * **Random Forest:** RMSE 5.28
  * **XGBoost:** RMSE 5.02
    XGBoost outperformed others, improving prediction accuracy by **11% compared to Linear Regression** and **5% compared to Random Forest**.

* **Business Insights via Power BI Dashboards:**

  * Demand and fare distributions were visualized by **hour of day, day of week, and passenger count**.
  * Dashboards highlighted that **pricing errors and prediction deviations were more frequent during late-night hours and weekends**.
  * Interactive exploration of trip patterns enables business teams to optimize fare policies and improve **customer trust through fairer pricing**.

---

### 5. Tech Stack

* **Python (Data Analysis & Machine Learning):**
  * **pandas & numpy** → data cleaning, transformation, and feature engineering.
  * **scikit-learn** → train-test split, evaluation metrics, and baseline models.
  * **XGBoost** → advanced gradient boosting for higher prediction accuracy.

* **Power BI (Business Intelligence & Visualization):**
  * Built interactive dashboards to analyze demand, fare patterns, and model performance.
  * Enabled stakeholders to drill down by **time of day, passenger count, and trip distance**.

* **Supporting Tools:**
  * PyCharm for development and experimentation.
  * GitHub for version control and project documentation.

---

### 6. End-to-End Capabilities Showcased

* **Data Cleaning & Feature Engineering**
  The project began with over 1 million raw taxi trip records that contained missing values, inconsistent entries, and potential outliers. I applied systematic data cleaning techniques, including handling null values, removing anomalies, and standardizing formats. Beyond cleaning, I engineered new features such as trip distance (via the haversine formula), log-transformed variables, and time-based attributes (hour, day of week, weekend/night indicators). These steps improved both model interpretability and predictive accuracy by enriching the dataset with meaningful predictors.

* **Machine Learning Modeling & Evaluation**
  Multiple machine learning approaches were implemented and compared to ensure robust predictions. Starting with Linear Regression as a baseline, I advanced to ensemble techniques such as Random Forest and XGBoost. Models were evaluated using industry-standard metrics including **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)**. Results demonstrated that XGBoost consistently outperformed other methods, achieving the lowest error rates and offering strong generalization capability. The comparison process highlighted not just technical proficiency but also the ability to justify model selection with evidence-based reasoning.

* **Business-Oriented Visualization**
  To bridge the gap between technical outputs and actionable business insights, I developed interactive dashboards in Power BI. These visualizations enabled exploration of fare distributions, demand fluctuations across time periods, passenger behavior patterns, and model error analysis. By designing user-friendly dashboards, the project demonstrated how predictive analytics can be made accessible to non-technical stakeholders, allowing decision-makers to interactively query the data and derive insights without coding knowledge.

* **Complete Documentation & Project Delivery**
  The entire workflow—from data ingestion and exploratory analysis to modeling and visualization—was documented in a structured manner. This ensured reproducibility and transparency, key qualities for real-world analytics projects. The documentation not only explains technical choices but also frames them in a business context, showing how machine learning and BI can be combined to solve operational challenges. This reflects a holistic approach to project delivery, where technical depth and business relevance go hand in hand.


---




