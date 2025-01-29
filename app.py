# library used to access directories and the .env file (NOT SHOWN for security reasons)
import os
from statistics import mode

# libraries to visualize and organize data
import numpy as np

import pandas as pd

# plotting utilities
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors

# for styling clusters made in the scatter plot (also a plot utility)
import seaborn as sns

# libraries to treat categorical variables within the dataframe
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

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

# fig, ax = plt.subplots()
#plt.boxplot([x_data,y_data], label=['X Data','Y Data'])
#plt.savefig("axisboxplots.png")

# plt.title("Deviation graph of likelihood for getting diabetes")
# plt.xlabel("Sums of Age, BMI, Fasting Blood Pressure, Smoking Status, Alcohol Intake, and Vitamin D Level")
# plt.ylabel("Frequencies of the Sums")
# plt.hist(x=x_data, label='X Data')
# plt.hist(x=y_data, label='Y Data')
# plt.savefig("diabeteshistogram.png")

plt.xlabel("Sum of Age, BMI, and Fasting Blood Pressure")
plt.ylabel("Sum of Smoking Status, Alcohol Intake, and Vitamin D Level")
plt.scatter(x=x_data, y=y_data)
plt.savefig("idscatter.png")


# Train and Test the ML Model
x_data = np.array(x_data).reshape(-1, 1)
y_data = np.array(y_data).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
y_train_norm = scaler.fit_transform(y_train)
y_test_norm = scaler.transform(y_test)

# Apply KMeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(X_train_norm)  # Get cluster labels

# These lines create the scatterplot
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    x=X_train_norm[:, 0], 
    y=y_train[:, 0], 
    hue=clusters, 
    palette=sns.color_palette("tab10", n_clusters),  # Ensure distinct colors
    edgecolor="k"
)

# Create a colorbar for the legend
handles, labels = scatter.get_legend_handles_labels()
plt.legend(title="Cluster Label", handles=handles, labels=labels)

# Titles and labels
plt.title("Scatterplot of Normalized Training Data with Clusters")
plt.xlabel("Normalized X Data")
plt.ylabel("Y Data")

# Save and show plot
plt.savefig("normalized_training_scatter.png")
