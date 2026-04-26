# 🚗 Car Price Prediction ML Web App

An end-to-end Machine Learning web application that predicts the selling price of used cars based on various features like brand, year, fuel type, mileage, and more.

---

## 🔗 Live Demo
> Run locally using the steps below

---

## 📌 Features

- 🔍 **Car Price Prediction** — Predicts used car price based on 11 input features
- 📉 **Depreciation Calculator** — Calculates car value after N years using brand-specific depreciation rates
- 🎛️ **Interactive UI** — Built with Streamlit for real-time user input and instant results
- 🧠 **ML Model** — Linear Regression trained on 8000+ real car listings

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | ML model training (Linear Regression) |
| Matplotlib & Seaborn | Data visualization & EDA |
| Streamlit | Web app framework |
| Pickle | Model serialization |

---

## 📂 Project Structure

```
car-price-prediction/
├── app.py                  # Streamlit web application
├── model.pkl               # Trained ML model
├── Car_details__1_.csv     # Dataset (8000+ car listings)
└── README.md               # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/dhruvsikka97-boop/car-price-prediction.git
cd car-price-prediction
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📊 Dataset

- **Source:** Car details dataset with 8,128 records
- **Features used:** Brand, Year, KMs Driven, Fuel Type, Seller Type, Transmission, Owner Type, Mileage, Engine CC, Max Power, Seats
- **Target:** Selling Price (₹)

---

## 🧪 Model Details

| Detail | Value |
|---|---|
| Algorithm | Linear Regression |
| Train/Test Split | 80% / 20% |
| Library | Scikit-learn |
| Saved As | model.pkl (Pickle) |

---

## 👨‍💻 Developer

**Dhruv Sikka**
- 📧 Dhruvsikka97@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/dhruv-sikka-622737262/)
- 🐙 [GitHub](https://github.com/dhruvsikka97-boop)

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
