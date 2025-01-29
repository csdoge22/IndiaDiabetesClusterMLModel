# library used to access directories and the .env file (NOT SHOWN for security reasons)
import os

# libraries to visualize and organize data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# libraries to treat categorical variables within the dataframe
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# libraries to train the model
from sklearn.model_selection import train_test_split
# libraries to load environment variables
from dotenv import load_dotenv

matplotlib.use('Agg')


"""Process data from the data set"""
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
diabetes_csv = pd.read_csv(os.path.join(BASE_DIR,"diabetes_prediction_india.csv"))
diabetes_csv.reset_index(drop=True, inplace=True)

"""TODO: Clean Up Data """
diabetes_csv = diabetes_csv.drop_duplicates().dropna()
diabetes_csv.drop('Diabetes_Status', axis=1, inplace=True)

"""TODO: Treat Categorical Variables """

# Initialize a dictionary to store LabelEncoders for each column
label_encoders = {}

# Iterate over all columns in the DataFrame
for col in diabetes_csv.columns:
    # Check if the column is categorical (dtype == object)
    if diabetes_csv[col].dtype == 'object':
        # Create a LabelEncoder for the column
        le = LabelEncoder()
        # Fit and transform the column
        diabetes_csv[col] = le.fit_transform(diabetes_csv[col])
        # Store the LabelEncoder for future use (if needed)
        label_encoders[col] = le

diabetes_csv.to_csv("NumerifiedData.csv", index="Index")
corr_matrix = diabetes_csv.corr()

# Create the heatmap
# plt.figure(figsize=(10, 8))  # Set the figure size
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)

# Add titles and labels
# plt.title("Correlation Heatmap", fontsize=16)

# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.yticks(rotation=0)
# plt.tight_layout()  # Adjust layout to prevent label cutoff
# plt.show()

# Have only the necessary columns for processing
# x_axis = np.array(diabetes_csv['Age'])
# y_axis = np.array(diabetes_csv['Smoking_Status'])

# print(x_axis)
# print(y_axis)


# plt.scatter(x=x_axis,y=y_axis)


# Print the processed DataFrame
diabetes_csv.reset_index(drop=True, inplace=True)

print(diabetes_csv)
"""TODO: Train and Test the Model """
df_age = diabetes_csv['Age'].to_numpy()
df_bmi = diabetes_csv['BMI'].to_numpy()
df_fbs = diabetes_csv['Fasting_Blood_Sugar'].to_numpy()
x_data = []
for i in range(len(df_age)):
    x_data.append(df_age[i]+ df_bmi[i]+df_fbs[i])

df_smoking = diabetes_csv['Smoking_Status'].to_numpy()
df_alcohol = diabetes_csv['Alcohol_Intake'].to_numpy()
df_vdl = diabetes_csv['Vitamin_D_Level'].to_numpy()
y_data = []

for i in range(len(df_smoking)):
    y_data.append(df_smoking[i]+ df_alcohol[i]+df_vdl[i])
print(y_data)

fig, ax = plt.subplots()
#plt.boxplot([x_data,y_data], label=['X Data','Y Data'])
#plt.savefig("axisboxplots.png")

# plt.hist(x=x_data, label='X Data')
# plt.hist(x=y_data, label='Y Data')
# plt.savefig("diabeteshistogram.png")

plt.scatter(x=x_data, y=y_data)
plt.savefig("idscatter.png")


# y_data = np.array(diabetes_csv['Smoking_Status'])

# for i in range(len(y_data)):
#     y_data[i] = diabetes_csv['Smoking_Status'].loc[i]+diabetes_csv['Alcohol_Intake'].loc[i]+diabetes_csv['Stress_Level'].loc[i]

# We will omit fields that are indirectly proportional to an increased risk of diabetes
# plt.scatter(x_data,y_data)


# we will try training with less and incrementing more until the ML model is accurate
# X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.9)

