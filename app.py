import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# library to load environment variables
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
diabetes_csv = pd.read_csv(os.path.join(BASE_DIR,"diabetes_prediction_india.csv"))

"""TODO: Clean Up Data """
diabetes_csv = diabetes_csv.drop_duplicates().dropna()

# print(diabetes_csv)
print(diabetes_csv.dtypes)
"""TODO: Train and Test the Model """