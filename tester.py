
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, HuberRegressor  # Removed models causing errors
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
# Removed GaussianProcessRegressor due to sparse data limitations
from sklearn.neural_network import MLPRegressor
from scipy.sparse import issparse

# Input data (assuming you have a CSV file named "Delhi.csv")
file_path = "G:\PYTHON\BTP PHASE 2\Real-Estate-Price-Prediction\Delhi.csv"
processed_data = pd.read_csv(file_path)

# Select the second row as input
input_data = processed_data.iloc[1, :]  # Extract 2nd row as Series

# Print the input data
print("Input data:")
print(input_data)

# Convert input data to DataFrame (if necessary)
if not isinstance(input_data, pd.DataFrame):
    columns = ["Price", "Area", "Location", "No. of Bedrooms", "Resale", "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens", "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall", "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security", "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine", "Gasconnection", "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant", "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator"]
    input_df = pd.DataFrame(input_data.values.reshape(1, -1), columns=columns)
else:
    input_df = input_data

# Define features and target variable
X = processed_data.drop(columns=['Price'])
y = processed_data['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if X_train is sparse and convert to dense format if necessary
if issparse(X_train):
    X_train_dense = X_train.toarray()
else:
    X_train_dense = X_train

# Define categorical and numerical features
categorical_features = ["Location"]
numerical_features = X_train.columns.drop(categorical_features)

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Initialize regression models (excluding models causing errors)
regressors = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "HuberRegressor": HuberRegressor(),
    "SVR": SVR(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    # Removed GaussianProcessRegressor due to sparse data limitations
    "MLPRegressor": MLPRegressor()
}

# Create a dictionary to store predictions
predictions = {}

# Fit and predict using each model
for name, regressor in regressors.items():
    # Create a pipeline with preprocessing and the current regressor
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', regressor)])
    # Fit the model using the dense data (if necessary)
    if issparse(X_train):
        model.fit(X_train_dense, y_train)
    else:
        model.fit(X_train, y_train)
    # Make predictions on the input data
    predictions[name] = model.predict(input_df)

# Print predictions
for name, prediction in predictions.items():
    print(f"{name}: {prediction}")