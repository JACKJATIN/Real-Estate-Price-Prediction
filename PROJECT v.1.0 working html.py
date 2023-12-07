import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from flask import Flask, render_template, request


# Function to read data from CSV
def wrangle(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# Load the data
file_path = "./Delhi.csv"  # Update with your file path
processed_data = wrangle(file_path)

# Define the target variable
target = "Price"
y_train = processed_data[target]

# Define the features including 'Location'
features = ["Area", "No. of Bedrooms", "Resale", "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
            "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall", "Intercom",
            "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security", "PowerBackup", "CarParking",
            "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine", "Gasconnection",
            "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant", "Microwave",
            "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator", "Location"]

# Extract the features
X_train = processed_data[features]

# One-hot encode the 'Location' feature
X_train = pd.get_dummies(X_train, columns=['Location'], drop_first=True)

# Define columns to impute
columns_to_impute = ["Area", "No. of Bedrooms", "Resale", "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
                     "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall", "Intercom",
                     "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security", "PowerBackup", "CarParking",
                     "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine", "Gasconnection",
                     "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant", "Microwave",
                     "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator"]

# Replace 9 with NaN for imputation
for column in columns_to_impute:
    X_train.loc[X_train[column] == 9, column] = np.nan

# Imputation using SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
X_train_imputed = imputer.fit_transform(X_train)

# Create dictionary of models
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function='MAE', verbose=0),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_train_imputed)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    print(f"MAE for {model_name}: {mae:.2f}")
    print(f"MSE for {model_name}: {mse:.2f}")
    print(f"RMSE for {model_name}: {rmse:.2f}")
    print("-" * 30)

# Initialize Flask app
app = Flask(__name__)

# Define function for making predictions
def make_prediction(data):
    for column in data.columns:
        data[column] = data[column].replace(9, np.nan)

    data = pd.get_dummies(data, columns=['Location'], drop_first=True)
    data = data.reindex(columns=X_train.columns, fill_value=0)
    data_imputed = imputer.transform(data)
    prediction = model.predict(data_imputed)[0]
    return prediction

# Route to render UI template with the form
@app.route('/')
def home():
    locations = processed_data['Location'].unique().tolist()
    return render_template('index.html', locations=locations)
    
# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Area": float(request.form['area']),
        "No. of Bedrooms": int(request.form['bedrooms']),
        "Resale": int(request.form['Resale']),
        "MaintenanceStaff": int(request.form['maintenance']),
        "Gymnasium": int(request.form['gymnasium']),
        "SwimmingPool": int(request.form['SwimmingPool']),
        "LandscapedGardens": int(request.form['LandscapedGardens']),
        "JoggingTrack": int(request.form['jogging_track']),
        "RainWaterHarvesting": int(request.form['rainwater_harvesting']),
        "IndoorGames": int(request.form['indoor_games']),
        "ShoppingMall": int(request.form['shopping_mall']),
        "Intercom": int(request.form['intercom']),
        "SportsFacility": int(request.form['sports_facility']),
        "ATM": int(request.form['atm']),
        "ClubHouse": int(request.form['club_house']),
        "School": int(request.form['school']),
        "24X7Security": int(request.form['security']),
        "PowerBackup": int(request.form['power_backup']),
        "CarParking": int(request.form['car_parking']),
        "StaffQuarter": int(request.form['staff_quarter']),
        "Cafeteria": int(request.form['cafeteria']),
        "MultipurposeRoom": int(request.form['multipurpose_room']),
        "Hospital": int(request.form['hospital']),
        "WashingMachine": int(request.form['washing_machine']),
        "Gasconnection": int(request.form['gas_connection']),
        "AC": int(request.form['ac']),
        "Wifi": int(request.form['wifi']),
        "Children'splayarea": int(request.form['play_area']),
        "LiftAvailable": int(request.form['lift_available']),
        "BED": int(request.form['bed']),
        "VaastuCompliant": int(request.form['vaastu_compliant']),
        "Microwave": int(request.form['microwave']),
        "GolfCourse": int(request.form['golf_course']),
        "TV": int(request.form['tv']),
        "DiningTable": int(request.form['dining_table']),
        "Sofa": int(request.form['sofa']),
        "Wardrobe": int(request.form['wardrobe']),
        "Refrigerator": int(request.form['refrigerator']),
        "Location": request.form['location']
    }

    df = pd.DataFrame(data, index=[0])
    prediction = make_prediction(df)

    return render_template('result.html', prediction=prediction)

# Run the app on localhost
if __name__ == '__main__':
    app.run(debug=True)