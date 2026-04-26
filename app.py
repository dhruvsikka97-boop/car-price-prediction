import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st


st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('Car_details.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# ── Input widgets ──────────────────────────────────────────────────────────────
name         = st.selectbox('Select Car Brand', cars_data['name'].unique())
year         = st.slider('Car Manufactured Year', 1994, 2024)
km_driven    = st.slider('No of kms Driven', 11, 2360457)
fuel         = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type  = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner        = st.selectbox('Owner type', cars_data['owner'].unique())   # ✅ Fixed label (was "Seller type")
mileage      = st.slider('Car Mileage (kmpl)', 10, 42)                   # ✅ Fixed max (was 40)
engine       = st.slider('Engine CC', 700, 3604)                         # ✅ Fixed max (was 5000)
max_power    = st.slider('Max Power (bhp)', 0, 400)                      # ✅ Fixed max (was 200)
seats        = st.slider('No of Seats', 2, 14)                           # ✅ Fixed min/max (was 5-10)

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner,
          mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type',
                 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True
    )
    input_data_model['fuel'].replace(
        ['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True
    )
    input_data_model['seller_type'].replace(
        ['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True
    )
    input_data_model['transmission'].replace(
        ['Manual', 'Automatic'], [1, 2], inplace=True
    )
    # ✅ Fixed: Added 'Peugeot' (brand 32) which exists in the dataset but was missing
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        inplace=True
    )

    car_price = model.predict(input_data_model)
    st.markdown('Car Price is going to be ₹' + str(round(car_price[0], 2)))


# ── Depreciation Calculator ────────────────────────────────────────────────────
st.header("Car Depreciation Calculator")

# ✅ Fixed: Expanded brand list to cover all brands in dataset
depr_rates = {
    "Maruti": 0.10, "Hyundai": 0.10, "Tata": 0.09, "Honda": 0.10,
    "Toyota": 0.07, "Mahindra": 0.12, "Ford": 0.11, "Renault": 0.11,
    "Skoda": 0.10, "Volkswagen": 0.10, "BMW": 0.15, "Audi": 0.15,
    "Mercedes-Benz": 0.14, "Kia": 0.10, "MG": 0.11, "Jeep": 0.10,
    "Nissan": 0.11, "Datsun": 0.12, "Chevrolet": 0.12, "Fiat": 0.13,
    "Mitsubishi": 0.11, "Volvo": 0.13, "Jaguar": 0.15, "Land": 0.13,
    "Lexus": 0.13, "Daewoo": 0.13, "Force": 0.12, "Ambassador": 0.14,
    "Ashok": 0.12, "Isuzu": 0.11, "Opel": 0.13, "Peugeot": 0.12,
}

dep_brand = st.selectbox("Select Car Brand for Depreciation", sorted(depr_rates.keys()))
dep_price = st.number_input("Enter Original Price (in ₹)", min_value=10000, step=10000)
dep_years = st.number_input("Enter Number of Years", min_value=1, max_value=20, step=1)

if st.button("Calculate Depreciation"):
    rate = depr_rates[dep_brand]
    current_value = dep_price * ((1 - rate) ** dep_years)
    st.success(f"Depreciated Value After {dep_years} Years: ₹{round(current_value, 2)}")
    st.info(f"Depreciation Rate used for {dep_brand}: {int(rate * 100)}% per year")
