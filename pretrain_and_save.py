import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor  # Import Huber Regressor
import pickle
from catboost import CatBoostRegressor
# Load the data
file_path = 'Delhi.csv'
processed_data = pd.read_csv(file_path)

# Define the target variable
target = "Price"
y_train = processed_data[target]

# Define the features including 'Location'
features = [
    "Area", "No. of Bedrooms", "Resale", "MaintenanceStaff", "Gymnasium",
    "SwimmingPool", "LandscapedGardens", "JoggingTrack", "RainWaterHarvesting",
    "IndoorGames", "ShoppingMall", "Intercom", "SportsFacility", "ATM",
    "ClubHouse", "School", "24X7Security", "PowerBackup", "CarParking",
    "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine",
    "Gasconnection", "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED",
    "VaastuCompliant", "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa",
    "Wardrobe", "Refrigerator", "Location"
]

# Extract the features
X_train = processed_data[features]

# One-hot encode the 'Location' feature
X_train = pd.get_dummies(X_train, columns=['Location'], drop_first=True)

# Define columns to impute
columns_to_impute = features[:-1]  # Exclude 'Location' as it's already encoded

# Replace 9 with NaN for imputation
for column in columns_to_impute:
    X_train.loc[X_train[column] == 9, column] = np.nan

# Imputation using SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
X_train_imputed = imputer.fit_transform(X_train)

# Train the models
models = {
    "Decision Tree": DecisionTreeRegressor().fit(X_train_imputed, y_train),
    "Random Forest": RandomForestRegressor().fit(X_train_imputed, y_train),
    "KNN": KNeighborsRegressor(n_neighbors=6).fit(X_train_imputed, y_train),
    "Huber Regressor": HuberRegressor().fit(X_train_imputed, y_train),
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function='MAE', verbose=0).fit(X_train_imputed, y_train),

}

# Define a location to save the models
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save each model using pickle
for model_name, model in models.items():
    model_file_path = os.path.join(models_dir, f"{model_name}.pkl")
    pickle.dump(model, open(model_file_path, 'wb'))

print("Models trained and saved successfully!")