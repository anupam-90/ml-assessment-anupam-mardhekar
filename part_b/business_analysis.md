#### B1. Problem Formulation



###### a) Problem Definition



This problem can be formulated as a supervised machine learning regression problem.



Target Variable:

items\_sold (number of items sold per store per month)

Input Features:

Store attributes: store\_size, location\_type, competition\_density

Promotion details: promotion\_type

Temporal features: month, year, weekend flag, festival flag

Customer behavior indicators: footfall, basket size (if available)

Problem Type:

Regression



Justification:

The goal is to predict a continuous numerical value (items sold). Since historical labelled data is available (past promotions and resulting sales), supervised learning is appropriate.



###### b) Why Items Sold Instead of Revenue



Using items sold (sales volume) is more reliable than revenue because:



Revenue can be distorted by price variations and discounts

Promotions like discounts reduce revenue per item but increase volume

The business objective is to understand promotion effectiveness, not pricing strategy



Broader Principle:

The target variable should directly reflect the business objective. Choosing the wrong target can mislead the model and result in poor decision-making.



###### c) Alternative Modelling Strategy



Instead of a single global model, a better approach is:



\- Segmented or Hierarchical Modelling



Options:



Build separate models for urban, semi-urban, and rural stores

Use store-level features or store embeddings

Apply mixed-effects models or cluster-based modelling



Justification:

Different stores respond differently to promotions due to demographics, competition, and customer behavior. A single model may average out these effects and reduce accuracy.



#### **B2. Data and EDA Strategy**

###### a) Data Joining Strategy



The four datasets should be joined as follows:



transactions table → base table

Join with:

store attributes using store\_id

promotion details using promotion\_type

calendar using transaction\_date



Final Dataset Grain:

&#x20;One row = store-month level observation



Aggregations:



Total items sold per store per month

Average basket size

Total number of transactions

Promotion applied in that month

Average competition density (if dynamic)



###### b) EDA Plan

Promotion vs Sales Analysis

Bar plot of average items sold by promotion type

\-Helps identify which promotions perform best

&#x20;Time Series Trend

&#x20;Line plot of monthly sales

\-Detect seasonality and trends

Store Segmentation Analysis

Compare sales across location types

\- Understand geographic differences

Correlation Heatmap

Identify relationships between features

\-Helps in feature selection and detecting multicollinearity

Distribution Analysis

Histogram of items sold

\-Check skewness and need for transformation



Impact on Modelling:



Helps decide feature engineering (e.g., seasonality features)

Identifies outliers and anomalies

Guides model selection and tuning



###### c) Handling Promotion Imbalance



Since 80% of transactions have no promotion:



Issues:



Model may become biased toward "no promotion"

Poor learning of promotion effects



Solutions:



Use resampling techniques (oversample promotion data)

Apply stratified sampling

Use feature weighting

Evaluate performance separately for promotion vs non-promotion cases



#### B3. Model Evaluation and Deployment

###### 

###### a) Train-Test Strategy



Use a time-based split:



Train on first \~2.5 years

Test on last \~6 months



Why not random split?



Breaks temporal order

Causes data leakage

Unrealistic evaluation



Evaluation Metrics:



RMSE (Root Mean Squared Error):

Penalizes large errors → important for business risk

MAE (Mean Absolute Error):

Easy to interpret → average error in units sold



Interpretation:



Lower RMSE → fewer large mistakes

Lower MAE → better overall prediction accuracy



###### b) Explaining Model Recommendations



Feature importance helps explain why different promotions are recommended:



Steps:



Check top influencing features (e.g., month, store type, competition)

Compare feature values for December vs March

Identify seasonal effects (festivals, holidays)

Use tools like:

Feature importance

SHAP values (optional advanced)



Communication to Business Team:



"In December, high demand due to festive season makes loyalty rewards effective"

"In March, lower demand makes discounts more impactful"

\### (c) Deployment Strategy



1\. Model Saving



Save trained pipeline using:

import joblib

joblib.dump(model, 'model.pkl')



2\. Monthly Prediction Pipeline



Collect new monthly data

Apply same preprocessing steps

Load saved model and generate predictions



3\. Automation



Schedule pipeline using cron/job scheduler



4\. Monitoring



Track:



Prediction vs actual error (RMSE, MAE)

Data drift (feature distribution changes)

Concept drift (performance degradation)



5\. Retraining Trigger



If error increases beyond threshold

Periodic retraining (e.g., every 3–6 months)

