import joblib
import os

# ========== Load trained model ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "rent_predictor_tuned.joblib")

if not os.path.exists(model_path):
    print("❌ Model not found. Please train the model first.")
    exit()

model = joblib.load(model_path)
print("✅ Model loaded successfully.")

# ========== Categorical Mappings (same as training) ==========
furnished_mapping = {"Unfurnished": 0, "Partly Furnished": 1, "Fully Furnished": 2}
region_mapping = {"Selangor": 0, "Kuala Lumpur": 1}
property_type_mapping = {
    "Apartment": 0, "Condominium": 1, "Flat": 2,
    "Service Residence": 3, "Others": 4
}
location_mapping = {
    "Petaling Jaya": 0, "Klang": 1, "Cheras": 2, "Subang Jaya": 3,
    "Shah Alam": 4, "Puchong": 5, "Ampang": 6, "Others": 7
}

# ========== Get user input ==========
print("\n=== Apartment Rent Prediction ===")
try:
    rooms = int(input("Enter number of rooms: "))
    size = float(input("Enter apartment size (in sqft): "))
    
    furnished_str = input("Furnishing (Unfurnished / Partly Furnished / Fully Furnished): ").strip()
    region_str = input("Region (Selangor / Kuala Lumpur): ").strip()
    property_type_str = input("Property Type (Apartment / Condominium / Flat / Service Residence / Others): ").strip()
    location_str = input("Location (Petaling Jaya / Klang / Cheras / Subang Jaya / Shah Alam / Puchong / Ampang / Others): ").strip()

    # Map input
    furnished = furnished_mapping.get(furnished_str, 0)
    region = region_mapping.get(region_str, 0)
    property_type = property_type_mapping.get(property_type_str, 4)
    location = location_mapping.get(location_str, 7)

except ValueError:
    print("❌ Invalid input. Please enter numeric values where needed.")
    exit()

# ========== Predict ==========
X_input = [[rooms, size, furnished, region, property_type, location]]
predicted_rent = model.predict(X_input)[0]

print(f"\n💰 Estimated Monthly Rent: RM {round(predicted_rent, 2)}")
