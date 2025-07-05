# 🏢 Mudah Apartment Rent Price Predictor

A machine learning project to predict **monthly rental prices** for apartments in **Kuala Lumpur and Selangor**, using data scraped from [mudah.my](https://www.mudah.my/). Built with Python and powered by a **Random Forest Regressor**, this project is designed for beginner-friendly learning and deployment.

---

## 📦 Project Structure
```
mudah-apartment-price-predictor/
├── data/
│ └── mudah-apartment-kl-selangor.csv # Raw dataset (scraped)
├── models/
│ └── rent_predictor_tuned.joblib # Trained ML model
├── scripts/
│ ├── train_model_gridsearch.py # Training script with GridSearchCV
│ └── predict.py # CLI-based prediction interface
├── .gitignore
├── README.md
├── requirements.txt
└── venv/ (excluded)
```

---

## 🚀 How It Works

### 🔢 Features Used
- `rooms`: Number of rooms
- `size`: Apartment size (in sqft)
- `furnished`: Furnishing status (Fully/Partly/Unfurnished)
- `region`: Selangor or Kuala Lumpur
- `property_type`: Apartment, Condo, Flat, etc.
- `location`: Petaling Jaya, Klang, Cheras, etc.

### 🎯 Target
- `monthly_rent`: Rent in RM per month

---

## 🛠️ How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mudah-apartment-price-predictor.git
cd mudah-apartment-price-predictor
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python scripts/train_model_gridsearch.py
```
- This will:
  - Clean the dataset
  - Encode categorical features
  - Train a Random Forest model with Grid Search
  - Save the trained model to models/

### 5. Predict Rent Price
```bash
python scripts/predict.py
```
- You will be prompted to enter apartment details (rooms, size, furnishing, region, etc.), and the model will return an estimated rent price in RM.
---
## 📊 Model Performance
Using RandomForestRegressor with Grid Search:

- ✅ RMSE: ~115
- ✅ R² Score: ~0.49
```
Note: Performance may vary based on data quality and feature engineering.
```
---
## 🧠 Skills Covered
- ✅ Data Cleaning
- ✅ Categorical Encoding
- ✅ Outlier Handling
- ✅ Train-Test Split
- ✅ Model Training (Random Forest)
- ✅ Grid Search & Hyperparameter Tuning
- ✅ Model Evaluation (RMSE, R²)
---
## 📌 Future Improvements
- Deploy using Streamlit or Gradio
- Add web scraping pipeline
- Add automated retraining
- Visualize predictions with charts
- Build a dashboard for real-time rent insights
