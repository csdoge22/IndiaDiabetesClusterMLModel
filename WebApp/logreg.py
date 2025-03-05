import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns

import os

from dotenv import load_dotenv

from sklearn.preprocessing import Normalizer, OneHotEncoder, PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline

load_dotenv()
matplotlib.use('Agg')
BASE_DIR = os.getenv('BASE_DIR')
diabetes_csv = pd.read_csv(os.path.join(BASE_DIR, "diabetes_prediction_india.csv"))

categorical_cols = ['Smoking_Status', 'Alcohol_Intake']
    
ohe = OneHotEncoder(sparse_output=False, drop="first")  # Avoid dummy variable trap
encoded_cats = ohe.fit_transform(diabetes_csv[categorical_cols])

encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(categorical_cols))

numerical_cols = ['Age', 'BMI', 'Fasting_Blood_Sugar', 'Vitamin_D_Level']
numerical_data = diabetes_csv[numerical_cols].to_numpy()

X = np.hstack((numerical_data, encoded_cats))
y = diabetes_csv['Diabetes_Status'].replace({'Positive': 1, 'Negative': 0}).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("Training Accuracy:", log_reg.score(X_train, y_train))
print("Test Accuracy:", log_reg.score(X_test, y_test))

plt.scatter(X_test[:, 0], y_test, label="Actual")
plt.scatter(X_test[:, 0], log_reg.predict(X_test), label="Predicted", alpha=0.5)
plt.xlabel("Age (example feature)")
plt.ylabel("Diabetes Status")
plt.legend()
plt.savefig("log_reg_graph.png")
