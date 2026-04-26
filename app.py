import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.header('Car Price Prediction ML Model')

# ── Load & prepare data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    cars = pd.read_csv('Car_details.csv')
    cars = cars.drop(columns=['torque'], errors='ignore')
    cars = cars.dropna().drop_duplicates()

    cars['name'] = cars['name'].apply(lambda x: x.split(' ')[0].strip())

    for col in ['mileage', 'engine', 'max_power']:
        cars[col] = cars[col].apply(lambda x: float(str(x).split()[0]) if str(x).split()[0] not in ['', 'nan'] else np.nan)

    cars = cars.dropna()

    cars['owner']        = cars['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':5})
    cars['fuel']         = cars['fuel'].map({'Diesel':1,'Petrol':2,'LPG':3,'CNG':4})
    cars['seller_type']  = cars['seller_type'].map({'Individual':1,'Dealer':2,'Trustmark Dealer':3})
    cars['transmission'] = cars['transmission'].map({'Manual':1,'Automatic':2})
    cars['name']         = cars['name'].map({'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,'Ford':6,'Renault':7,
                                              'Mahindra':8,'Tata':9,'Chevrolet':10,'Datsun':11,'Jeep':12,'Mercedes-Benz':13,
                                              'Mitsubishi':14,'Audi':15,'Volkswagen':16,'BMW':17,'Nissan':18,'Lexus':19,
                                              'Jaguar':20,'Land':21,'MG':22,'Volvo':23,'Daewoo':24,'Kia':25,'Fiat':26,
                                              'Force':27,'Ambassador':28,'Ashok':29,'Isuzu':30,'Opel':31,'Peugeot':32})
    cars = cars.dropna()

    X = cars[['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']]
    y = cars['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

@st.cache_data
def load_data():
    cars = pd.read_csv('Car_details.csv')
    cars['name'] = cars['name'].apply(lambda x: x.split(' ')[0].strip())
    return cars

model    = load_model()
cars_data = load_data()

# ── Input widgets ──────────────────────────────────────────────────────────────
name         = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
year         = st.slider('Car Manufactured Year', 1994, 2024)
km_driven    = st.slider('No of kms Driven', 11, 200000)
fuel         = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type  = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner        = st.selectbox('Owner type', cars_data['owner'].unique())
mileage      = st.slider('Car Mileage (kmpl)', 10, 42)
engine       = st.slider('Engine CC', 700, 3604)
max_power    = st.slider('Max Power (bhp)', 0, 400)
seats        = st.slider('No of Seats', 2, 14)

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("Predict"):
    owner_map        = {'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':5}
    fuel_map         = {'Diesel':1,'Petrol':2,'LPG':3,'CNG':4}
    seller_map       = {'Individual':1,'Dealer':2,'Trustmark Dealer':3}
    transmission_map = {'Manual':1,'Automatic':2}
    name_map         = {'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,'Ford':6,'Renault':7,
                        'Mahindra':8,'Tata':9,'Chevrolet':10,'Datsun':11,'Jeep':12,'Mercedes-Benz':13,
                        'Mitsubishi':14,'Audi':15,'Volkswagen':16,'BMW':17,'Nissan':18,'Lexus':19,
                        'Jaguar':20,'Land':21,'MG':22,'Volvo':23,'Daewoo':24,'Kia':25,'Fiat':26,
                        'Force':27,'Ambassador':28,'Ashok':29,'Isuzu':30,'Opel':31,'Peugeot':32}

    input_data = pd.DataFrame([[
        name_map.get(name, 1), year, km_driven, fuel_map.get(fuel, 1),
        seller_map.get(seller_type, 1), transmission_map.get(transmission, 1),
        owner_map.get(owner, 1), mileage, engine, max_power, seats
    ]], columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])

    car_price = model.predict(input_data)
    st.success(f'🚗 Estimated Car Price: ₹{round(car_price[0], 2):,}')

# ── Depreciation Calculator ────────────────────────────────────────────────────
st.header("Car Depreciation Calculator")

depr_rates = {
    "Maruti":0.10,"Hyundai":0.10,"Tata":0.09,"Honda":0.10,"Toyota":0.07,
    "Mahindra":0.12,"Ford":0.11,"Renault":0.11,"Skoda":0.10,"Volkswagen":0.10,
    "BMW":0.15,"Audi":0.15,"Mercedes-Benz":0.14,"Kia":0.10,"MG":0.11,
    "Jeep":0.10,"Nissan":0.11,"Datsun":0.12,"Chevrolet":0.12,"Fiat":0.13,
    "Mitsubishi":0.11,"Volvo":0.13,"Jaguar":0.15,"Land":0.13,"Lexus":0.13,
    "Daewoo":0.13,"Force":0.12,"Ambassador":0.14,"Ashok":0.12,"Isuzu":0.11,
    "Opel":0.13,"Peugeot":0.12,
}

dep_brand = st.selectbox("Select Car Brand for Depreciation", sorted(depr_rates.keys()))
dep_price = st.number_input("Enter Original Price (in ₹)", min_value=10000, step=10000)
dep_years = st.number_input("Enter Number of Years", min_value=1, max_value=20, step=1)

if st.button("Calculate Depreciation"):
    rate = depr_rates[dep_brand]
    current_value = dep_price * ((1 - rate) ** dep_years)
    st.success(f"💰 Depreciated Value After {dep_years} Years: ₹{round(current_value, 2):,}")
    st.info(f"📉 Depreciation Rate for {dep_brand}: {int(rate * 100)}% per year")
