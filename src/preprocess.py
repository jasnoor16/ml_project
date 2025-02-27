# Here we are importing necessary libraries for data processing
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib  # This is used to save encoders for later use in training and prediction


# Setting Up File Paths

# This is where our processed files will be saved. We creating the directory if it does not exist.
processed_dir = "./data/processed/"
os.makedirs(processed_dir, exist_ok=True)  

# Paths to cleaned data for both years (2023 and 2024)
CLEANED_2023_PATH = "./data/processed/cleaned_data_2023.csv"
CLEANED_2024_PATH = "./data/processed/cleaned_data_2024.csv"

# Defining the target variable (the column we want to predict)
TARGET = 'Donation Bags Collected'

# Cleaning Data

# Some columns are not needed for model training, so we are removing them.
DROP_COLUMNS_2023 = ['Date', 'Location', 'time completed', 'Route Number/Name']
DROP_COLUMNS_2024 = ['How did you receive the form?', 'Drop Off Location', 
                     'Route Number/Name', 'Additional Routes completed (2 routes)',
                     'Additional routes completed (3 routes)', 'Additional routes completed (3 routes)2',
                     'Additional routes completed (More than 3 Routes)',
                     'Additional routes completed (More than 3 Routes)2',
                     'Additional routes completed (More than 3 Routes)3',
                     'Comments or Feedback']

# The column names in 2024 data are slightly different from 2023, so we are renaming them to match
rename_2024 = {
    'No of Adult Volunteers': '# of Adult Volunteers',
    'No of Youth Volunteers': '# of Youth Volunteers',
    'Time to Complete (min)': 'Time Spent',
    'How many routes did you complete?': 'Routes Completed',
    'Ward': 'Ward/Branch',
    'Comments or Feedback': 'Comments'
}

# Loading the cleaned datasets into Pandas DataFrames
df_2023 = pd.read_csv(CLEANED_2023_PATH)
df_2024 = pd.read_csv(CLEANED_2024_PATH)

# Renaming 2024 dataset columns so they match 2023
df_2024.rename(columns=rename_2024, inplace=True)

# Removing the unnecessary columns we listed above
df_2023.drop(columns=DROP_COLUMNS_2023, errors='ignore', inplace=True)
df_2024.drop(columns=DROP_COLUMNS_2024, errors='ignore', inplace=True)

# Making sure that both datasets have the same columns (keeping only common columns)
common_columns = df_2023.columns.intersection(df_2024.columns)
df_2024 = df_2024[common_columns]

# Handling Categorical Values

# The column "Completed More Than One Route" contains "Yes" or "No", so we are converting them to 1 and 0
def convert_yes_no(df, column):
    if column in df.columns:  # Only apply if the column exists
        df[column] = df[column].map({'Yes': 1, 'No': 0})

convert_yes_no(df_2023, 'Completed More Than One Route')
convert_yes_no(df_2024, 'Completed More Than One Route')

# The column "Routes Completed" contains numbers as strings, so we are mapping them to actual numbers
route_mapping = {'1': 1, '2': 2, '3': 3, 'More than 3': 4}
df_2023['Routes Completed'] = df_2023['Routes Completed'].map(route_mapping)
df_2024['Routes Completed'] = df_2024['Routes Completed'].map(route_mapping)

# The "Time Spent" column has different numerical values, so we are categorizing them into meaningful ranges
def convert_time_spent(value):
    if value <= 30:
        return "0 - 30 Minutes"
    elif 30 < value <= 60:
        return "30 - 60 Minutes"
    elif 60 < value <= 90:
        return "1 Hour - 1.5 Hours"
    else:
        return "2+ Hours"

# Applying this transformation only to 2023 data, since we assume 2024 data is already formatted
df_2023['Time Spent'] = df_2023['Time Spent'].apply(convert_time_spent)

# Converting "Time Spent" categorical values into numeric labels (since models only work with numbers)
encoder = LabelEncoder()
df_2023['Time Spent'] = encoder.fit_transform(df_2023['Time Spent'])
df_2024['Time Spent'] = encoder.transform(df_2024['Time Spent'])  # Use the same transformation as 2023


# Encoding Categorical Variables

# Some columns like "Stake" and "Ward/Branch" contain text values. We need to convert them into numbers using one-hot encoding.
categorical_features = ['Stake', 'Ward/Branch']

# OneHotEncoder creates separate binary columns for each unique category
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Splitting Data into Features and Target

# The target column (Donation Bags Collected) is what we want to predict.
X_train = df_2023.drop(columns=[TARGET])  # Features for 2023
y_train = df_2023[TARGET]  # Target variable for 2023
X_test = df_2024.drop(columns=[TARGET])  # Features for 2024
y_test = df_2024[TARGET]  # Target variable for 2024

# Applying transformations to convert categorical variables to numbers
X_train_encoded = ct.fit_transform(X_train)
X_test_encoded = ct.transform(X_test)

# Converting all columns to numeric before handling missing values
X_train_encoded = pd.DataFrame(X_train_encoded).apply(pd.to_numeric, errors='coerce')
X_test_encoded = pd.DataFrame(X_test_encoded).apply(pd.to_numeric, errors='coerce')

# Handling Missing Values

# If a column has some missing values, we replace them with the median of that column
# If the entire column is missing (all values are NaN), we replace it with 0
X_train_encoded = X_train_encoded.apply(lambda col: col.fillna(col.median()) if not col.isna().all() else col.fillna(0), axis=0)
X_test_encoded = X_test_encoded.apply(lambda col: col.fillna(col.median()) if not col.isna().all() else col.fillna(0), axis=0)


# Feature Scaling

# Since machine learning models perform better with standardized data, we scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)  # Using the same transformation on test data

# Saving Preprocessed Data

# Saving the processed datasets for later use in training and evaluation
np.save(os.path.join(processed_dir, "X_train.npy"), X_train_scaled)
np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled)
np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

# Saving Encoders for Later Use

# Saving the transformations so that they can be applied to new data during model evaluation and prediction
joblib.dump(ct, os.path.join(processed_dir, "column_transformer.pkl"))
joblib.dump(scaler, os.path.join(processed_dir, "scaler.pkl"))
joblib.dump(encoder, os.path.join(processed_dir, "time_spent_label_encoder.pkl"))
