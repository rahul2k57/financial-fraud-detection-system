# Financial Fraud Detection System

A high-performance machine learning application designed to identify fraudulent transactions in real-time. By leveraging a Random Forest Classifier and advanced feature engineering, this system achieves near-perfect detection rates on the PaySim synthetic dataset.

### Key Performance Metrics:

**Model Accuracy:** 99.9%

**ROC-AUC Score:** 0.9979

**Recall (Fraud Detection):** ~99% (Minimized False Negatives)


### Technical Methodology:

### **Feature Engineering:**

Custom features were created to better detect suspicious transactions:

**errorBalanceOrig:** Identifies cases where the account balance does not match the expected value after a transaction—a common sign of unauthorized money transfers.

**isZeroDest:** Detects "mule account" behavior, where money is transferred to accounts that were previously inactive.

### Handling Class Imbalance (SMOTE):

In the PaySim dataset, fraudulent transactions make up less than 1% of the total data. To address this, the SMOTE (Synthetic Minority Over-sampling Technique) method was used. SMOTE generates synthetic fraud samples so the model learns patterns effectively instead of simply biasedly predicting "safe" for every transaction.

### Data Integrity & Leakage Prevention:

To ensure a fair and reliable evaluation, a strict data processing pipeline was followed:

Split First: The dataset was split into training and testing sets before any transformations.

Isolated Scaling: The StandardScaler was fitted only on the training data to prevent "future knowledge" leakage.

Training-Only SMOTE: SMOTE was applied only to the training set. Testing data remained unchanged to reflect real-world scenarios.


### Tech Stack and Tools:

**Python:** (Version 3.9 or higher)

**Scikit-Learn:** Model building and data scaling.

**Imbalanced-Learn:** SMOTE implementation for 1:1000 class ratios.

**Streamlit:** Real-time Web Interface.

**Pandas & NumPy:** Data manipulation and math.

**Joblib:** Saving/loading trained model artifacts (.pkl).

**Matplotlib & Seaborn:** EDA and performance visualization.


### How to Run Locally:

**Clone the repository:**

```
git clone https://github.com/rahul2k57/financial-fraud-detection-system.git
```
cd financial-fraud-detection-system

**Install dependencies:**

```
pip install -r requirements.txt
```
**Download the Dataset:**

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection) and and place it in the project's root directory.

Launch the Streamlit App:

```
streamlit run app.py
```
