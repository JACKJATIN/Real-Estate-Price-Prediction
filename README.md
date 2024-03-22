
# Real Estate Price Estimation 

Live Link 
[Click Here](https://realestate-price-estimator.onrender.com/)
A brief description of what this project does and who it's for

This project is a web application for predicting house prices based on various features. Users can input details such as area, number of bedrooms, and amenities, and the application will provide an estimated house price within India.

## Features

- Predicts house prices using a machine learning model.
- User-friendly web interface to input property details.
- Supports various amenities and features for accurate predictions.
- Using Google Maps Api mapping various GIS features that changes the price of a real estate location.

## Requirements

- Python 3.x
- Flask web framework
- Machine learning libraries (e.g., scikit-learn)
- HTML, CSS, and JavaScript (for the web interface)
## Installation 

To run this project follow the steps:

1. Clone the Directory
```bash
  git clone https://github.com/JACKJATIN/Real-Estate-Price-Prediction.git
```
2. Navigate to the project directory
```bash
   cd house-price-prediction
```    
3. Make A Virtual Enviroment
```bash
   python -m venv "YOUR VIRTUAL ENVIROMENT NAME"
```   
4. Install dependencies
```bash
   pip install -r \requirements.txt
```
## Getting a Google Maps API Key

1. Sign In or Sign Up for a Google Cloud Account: Go to the Google Cloud Console and sign in or create a new account.

2. Create a New Project: If you don't have a project yet, click on the project dropdown menu at the top of the page and select "New Project". Give your project a name and click "Create".

3. Enable the Maps JavaScript API: In the Google Cloud Console, navigate to the "APIs & Services" > "Library" section. Search for "Maps JavaScript API" and click on it. Then, click the "Enable" button.

4. Create API Key: After enabling the Maps JavaScript API, navigate to the "APIs & Services" > "Credentials" section. Click on "Create credentials" and select "API key". A new API key will be generated.

5. Restrict the API Key (Optional but recommended): For security reasons, it's a good practice to restrict the usage of your API key. You can restrict it by IP address, HTTP referrer, or by enabling specific APIs. You can set restrictions by clicking on your API key in the "Credentials" section.

6. Copy the API Key: Once you've created the API key, copy it. You'll need to replace "YOUR_GOOGLE_MAPS_API_KEY" with your actual API key in the project.

## **Important: Google Maps API Key Integration**

1.Replace the API Key in the Project: After obtaining your Google Maps API key, navigate to the project directory where you've cloned the repository.

2.Locate the Code Where the API Key is Needed: In the project files, there should be a section where the Google Maps API key is used. Look for any files or code sections where "YOUR_GOOGLE_MAPS_API_KEY" is mentioned.

3.Replace Placeholder with Your API Key: Replace "YOUR_GOOGLE_MAPS_API_KEY" with the API key you obtained in step 6.

4.Save Changes: Save the changes to the file.

## Usage
Pre-Train or Retrain data initially 
```bash
   python pretrain_and_save.py
```

Run the Flask application
```bash
   python app.py
```
Open your web browser and go to http://localhost:5000 to access the application.

Fill in the property details and click the "Predict" button to get the estimated house    price.

Feel Free to explore the UI.
