#key='YOUR_GOOGLE_MAPS_API_KEY'
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor  # Import Huber Regressor
from catboost import CatBoostRegressor
from flask import Flask, render_template, request, url_for
import googlemaps
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

# Function to read data from CSV
def wrangle(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# Load the data
file_path = 'Delhi.csv'  # Use raw string literal or forward slashes
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

# Initialize Flask app
app = Flask(__name__)

# Train the models
models = {
    "Decision Tree": DecisionTreeRegressor().fit(X_train_imputed, y_train),
    "Random Forest": RandomForestRegressor().fit(X_train_imputed, y_train),
    #"CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function='MAE', verbose=0).fit(X_train_imputed, y_train),
    "KNN": KNeighborsRegressor(n_neighbors=6).fit(X_train_imputed, y_train),
    "Huber Regressor": HuberRegressor().fit(X_train_imputed, y_train)  # Train Huber Regressor
}

def make_prediction(data):
    # One-hot encode the 'Location' feature
    data = pd.get_dummies(data, columns=['Location'], drop_first=True)
    
    # Align columns with training data
    data = data.reindex(columns=X_train.columns, fill_value=0)
    
    data_imputed = imputer.transform(data)

    predictions = {}
    for model_name, model in models.items():
        prediction = round(model.predict(data_imputed)[0], 2)  # Round off to 2 decimal places
        predictions[model_name] = prediction

    return predictions


# Route to render UI template with the form
@app.route('/')
def home():
    banner_image_url = url_for('static', filename='GOV_banner.png')
    return render_template('home.html', banner_image_url=banner_image_url)

@app.route('/inputs')
def inputs():
    locations = processed_data['Location'].unique().tolist()
    return render_template('inputs.html', locations=locations)

@app.route('/aboutus')
def about():
    team_members = {
        'Jatin Jangid': {
            'image_url': url_for('static', filename='profile-pic4.png'),
            'email': 'jatin.jangid.ug20@nsut.ac.in'
        },
        'Shreyansh Tiwari': {
            'image_url': url_for('static', filename='profile-pic3.png'),
            'email': 'shreyansh.tiwari.ug20@nsut.ac.in'
        },
        'Rishabh Jain': {
            'image_url': url_for('static', filename='profile-pic2.png'),
            'email': 'rishabh.jain.ug20@nsut.ac.in'
        },
        'Chandra Pratap': {
            'image_url':  url_for('static', filename='profile-pic1.png'),
            'email': 'chandra.pratap.ug20@nsut.ac.in'
        },
        'bg-image':url_for('static' , filename='10974.jpg')
    }
    return render_template('aboutus.html', team_members=team_members)
# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Area": float(request.form.get('area', 0)),
        "No. of Bedrooms": int(request.form.get('bedrooms', 0)),
        "Resale": int(request.form.get('Resale', 0)),
        "MaintenanceStaff": int(request.form.get('maintenance', 0)),
        "Gymnasium": int(request.form.get('gymnasium', 0)),
        "SwimmingPool": int(request.form.get('SwimmingPool', 0)),
        "LandscapedGardens": int(request.form.get('LandscapedGardens', 0)),
        "JoggingTrack": int(request.form.get('jogging_track', 0)),
        "RainWaterHarvesting": int(request.form.get('rainwater_harvesting', 0)),
        "IndoorGames": int(request.form.get('indoor_games', 0)),
        "ShoppingMall": int(request.form.get('shopping_mall', 0)),
        "Intercom": int(request.form.get('intercom', 0)),
        "SportsFacility": int(request.form.get('sports_facility', 0)),
        "ATM": int(request.form.get('atm', 0)),
        "ClubHouse": int(request.form.get('club_house', 0)),
        "School": int(request.form.get('school', 0)),
        "24X7Security": int(request.form.get('security', 0)),
        "PowerBackup": int(request.form.get('power_backup', 0)),
        "CarParking": int(request.form.get('car_parking', 0)),
        "StaffQuarter": int(request.form.get('staff_quarter', 0)),
        "Cafeteria": int(request.form.get('cafeteria', 0)),
        "MultipurposeRoom": int(request.form.get('multipurpose_room', 0)),
        "Hospital": int(request.form.get('hospital', 0)),
        "WashingMachine": int(request.form.get('washing_machine', 0)),
        "Gasconnection": int(request.form.get('gas_connection', 0)),
        "AC": int(request.form.get('ac', 0)),
        "Wifi": int(request.form.get('wifi', 0)),
        "Children'splayarea": int(request.form.get('play_area', 0)),
        "LiftAvailable": int(request.form.get('lift_available', 0)),
        "BED": int(request.form.get('bed', 0)),
        "VaastuCompliant": int(request.form.get('vaastu_compliant', 0)),
        "Microwave": int(request.form.get('microwave', 0)),
        "GolfCourse": int(request.form.get('golf_course', 0)),
        "TV": int(request.form.get('tv', 0)),
        "DiningTable": int(request.form.get('dining_table', 0)),
        "Sofa": int(request.form.get('sofa', 0)),
        "Wardrobe": int(request.form.get('wardrobe', 0)),
        "Refrigerator": int(request.form.get('refrigerator', 0)),
        "Location": request.form.get('location', '')
    }

    df = pd.DataFrame(data, index=[0])
    predictions = make_prediction(df)

    # Calculate average predicted price
    avg_predicted_price = round(sum(predictions.values()) / len(predictions), 2)

    # Geocode the predicted location to get its coordinates
    geocoder = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))  # Replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual API key
    geocode_result = geocoder.geocode(data['Location'])

    if geocode_result:
        latitude = geocode_result[0]['geometry']['location']['lat']
        longitude = geocode_result[0]['geometry']['location']['lng']
        
        # Render result.html with predictions and maps_html_content
        return render_template('result.html', latitude=latitude, longitude=longitude , location=data['Location'],predictions=predictions,avg_predicted_price=avg_predicted_price,key=os.getenv('GOOGLE_MAPS_API_KEY'))

    else:
        return "Location not found"
        
if __name__ == '__main__':
    app.run(debug=False)
