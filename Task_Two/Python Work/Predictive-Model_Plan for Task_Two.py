
pip install xgboost shap fairlearn scikit-learn
!pip install xgboost shap fairlearn scikit-learn

which python  # macOS/Linux
where python  # Windows

pip install --user xgboost shap fairlearn

conda install -c conda-forge shap fairlearn
pip install xgboost



# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb  # Should work now
import shap            # Should work now
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference  # Should work now

# Load and preprocess data
df = pd.read_csv("C:\\Users\\sande\\Downloads\\TATA's_DataAnalytics_certificate_by_forage\\Task_One\\Python work\\final saved and cleaned dataset with help of python\\geldium_data_cleaned.csv")

# Feature engineering
df['Payment_Consistency'] = df[['Month_1','Month_2','Month_3']].apply(
    lambda x: (x == 'On-time').mean(), axis=1)

# Encode categoricals
df['Employment_Status'] = df['Employment_Status'].map({
    'Employed':0, 'Self-Employed':1, 'Unemployed':2, 'Retired':3})

# Select features
X = df[['Credit_Utilization', 'Missed_Payments', 'Income', 
        'Payment_Consistency', 'Employment_Status']]
y = df['Delinquent_Account']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=5,  # Handle class imbalance
    max_depth=6,
    subsample=0.8
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_pred):.2f}")

# Explainability with SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)  # Visualize feature impacts

# Fairness check
demo_parity = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test['Employment_Status']
)
print(f"Fairness (Demographic Parity): {demo_parity:.2f}")
