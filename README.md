Financial Fraud Detection System
A high-performance machine learning application designed to identify fraudulent transactions in real-time. By leveraging a Random Forest Classifier and advanced feature engineering, this system achieves near-perfect detection rates on the PaySim synthetic dataset.

Key Performance Metrics:
Model Accuracy: 99.9%
ROC-AUC Score: 0.9979
Recall (Fraud Detection): ~99% (Minimized False Negatives)

Technical Methodology:
Feature Engineering:
Custom features were created to better detect suspicious transactions. For example, errorBalanceOrig helps identify cases where the account balance does not match the expected value after a transaction, which is a common sign of unauthorized money transfers. Another feature, isZeroDest, was introduced to detect mule account behavior, where money is transferred to accounts that were previously inactive.

Handling Class Imbalance (SMOTE):
In the PaySim dataset, fraudulent transactions make up less than 1% of the total data. Because of this imbalance, the SMOTE (Synthetic Minority Over-sampling Technique) method was used. SMOTE generates synthetic fraud samples so the model can learn fraud patterns more effectively instead of simply predicting most transactions as “safe”.

Data Integrity & Leakage Prevention:
To ensure a fair and reliable evaluation, a strict data processing pipeline was followed:

Step 1: The dataset was first split into training and testing sets before applying any transformations.
Step 2: The StandardScaler was fitted only on the training data to prevent the model from learning the distribution of the test set.
Step 3: SMOTE was applied only to the training set. The testing data remained unchanged and imbalanced, which better reflects real-world scenarios.

Evaluation Strategy:
Although the model achieved 99.9% accuracy, the focus was placed on more meaningful metrics such as ROC-AUC (0.9979) and Recall (99.3%). In financial fraud detection, missing a fraudulent transaction (False Negative) can cause serious losses. Therefore, the system is designed to prioritize detecting as many fraud cases as possible, even if it occasionally flags a legitimate transaction.

Tech Stack and Tools:
Python: Core programming language. (python 3.9 or higher version)
Scikit-Learn: Used for building the Random Forest model and data scaling.
Imbalanced-Learn (SMOTE): Essential for handling the 1:1000 fraud-to-safe transaction ratio.
Streamlit: Framework used to build the real-time Web Interface.
Pandas & NumPy: High-performance data manipulation and mathematical operations.
Joblib: Used for saving and loading the trained model artifacts (.pkl files).
Matplotlib & Seaborn: Used for Exploratory Data Analysis (EDA) and performance visualization.

 How to Run Locally:
Clone the repository:```bash
   git clone [https://github.com/rahul2k57/financial-fraud-detection-system.git](https://github.com/rahul2k57/financial-fraud-detection-system.git)
   cd financial-fraud-detection-system```

Install dependencies: ```bash
pip install -r requirements.txt```

Download the Dataset:
Download the `onlinefraud.csv` from [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection) and place it in the project's root directory.

Launch the Streamlit App:```bash
streamlit run app.py```
 
