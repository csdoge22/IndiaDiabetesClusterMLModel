import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns

import os

from dotenv import load_dotenv

from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
diabetes_csv = pd.read_csv(os.path.join(BASE_DIR,"diabetes_prediction_india.csv"))

diabetes_csv = diabetes_csv.drop_duplicates().dropna()
diabetes_csv.reset_index(drop=True, inplace=True)

fasting_blood_sugar = np.array(diabetes_csv['Fasting_Blood_Sugar'])
postprandial_blood_sugar = np.array(diabetes_csv['Postprandial_Blood_Sugar'])

average_blood_sugar = postprandial_blood_sugar
for i in range(len(fasting_blood_sugar)):
    average_blood_sugar[i]= (fasting_blood_sugar[i]+postprandial_blood_sugar[i])/2

x_data = fasting_blood_sugar.reshape(-1,1)
y_data = average_blood_sugar.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

def linear_regression():
    """Trains data in diabetes_prediction_india.csv for a linear regression model"""
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train)

    print("LINEAR REGRESSION ACCURACIES: ")
    print("TRAINING: ",lin_reg.score(X_train,y_train))
    print("TESTING: ",lin_reg.score(X_test,y_test))

    slope, intercept, r, p, std_err = stats.linregress(x_data, y_data)

    def myfunc(x):
        return slope * x + intercept

    linear_model = list(map(myfunc, x_data))
    
    stats.linregress()
    plt.xlabel(xlabel= "Age")
    plt.ylabel(ylabel="Average Blood Sugar")
    plt.scatter(x=x_data, y=y_data)
    plt.plot(x_data,linear_model)
    plt.savefig("age_sugar_correlation.png")

def polynomial_regression(degree: int):
    lin_reg = PolynomialFeatures(degree)
    lin_reg.fit(X_train,y_train)

    print("LINEAR REGRESSION ACCURACIES: ")
    print("TRAINING: ",lin_reg.score(X_train,y_train))
    print("TESTING: ",lin_reg.score(X_test,y_test))

    plt.xlabel(xlabel= "Age")
    plt.ylabel(ylabel="Average Blood Sugar")
    plt.scatter(x=x_data, y=y_data)
    plt.savefig("age_sugar_correlation.png")