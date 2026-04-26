import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoInsight Pro — Car Price & Depreciation",
    page_icon="🚗",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #08080f; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #0d0d1f 0%, #131328 100%);
    border: 1px solid #7c5cfc30;
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #7c5cfc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero p { color: #888; font-size: 1rem; margin: 0; }

.section-card {
    background: #0d0d1f;
    border: 1px solid #7c5cfc22;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0d7ff;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #7c5cfc40;
}

.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #13132a, #1a1a35);
    border: 1px solid #7c5cfc33;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; color: #7c5cfc; }
.metric-val.green { color: #5cf8a0; }
.metric-val.orange { color: #f8a05c; }
.metric-val.red { color: #f85c5c; }
.metric-lbl { font-size: 0.78rem; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

.result-banner {
    background: linear-gradient(135deg, #7c5cfc15, #5c7cfc15);
    border: 1px solid #7c5cfc44;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    text-align: center;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #7c5cfc;
}
.result-label { color: #aaa; font-size: 0.9rem; margin-top: 4px; }

.compare-col {
    background: #13132a;
    border: 1px solid #7c5cfc22;
    border-radius: 12px;
    padding: 1.2rem;
}
.compare-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #7c5cfc;
    margin-bottom: 1rem;
    text-align: center;
}

.insight-box {
    background: #0f1f15;
    border: 1px solid #5cf8a044;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    color: #5cf8a0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.warn-box {
    background: #1f1505;
    border: 1px solid #f8a05c44;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    color: #f8a05c;
    font-size: 0.9rem;
    line-height: 1.6;
}

.stButton>button {
    background: linear-gradient(135deg, #7c5cfc, #5c7cfc);
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.88; transform: translateY(-1px); }

label, .stSelectbox label, .stSlider label { color: #bbb !important; font-size: 0.88rem !important; }
.stSelectbox>div>div { background: #13132a !important; border-color: #7c5cfc33 !important; color: #eee !important; }

.tab-content { padding-top: 1rem; }

footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Data Cleaning Helper ───────────────────────────────────────────────────────
def clean_unit(val):
    try:
        parts = str(val).strip().split()
        return float(parts[0]) if parts else np.nan
    except:
        return np.nan

# ── Load & Train Model ─────────────────────────────────────────────────────────
NAME_MAP = {
    'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,'Ford':6,'Renault':7,
    'Mahindra':8,'Tata':9,'Chevrolet':10,'Datsun':11,'Jeep':12,'Mercedes-Benz':13,
    'Mitsubishi':14,'Audi':15,'Volkswagen':16,'BMW':17,'Nissan':18,'Lexus':19,
    'Jaguar':20,'Land':21,'MG':22,'Volvo':23,'Daewoo':24,'Kia':25,'Fiat':26,
    'Force':27,'Ambassador':28,'Ashok':29,'Isuzu':30,'Opel':31,'Peugeot':32
}
OWNER_MAP        = {'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':5}
FUEL_MAP         = {'Diesel':1,'Petrol':2,'LPG':3,'CNG':4}
SELLER_MAP       = {'Individual':1,'Dealer':2,'Trustmark Dealer':3}
TRANSMISSION_MAP = {'Manual':1,'Automatic':2}

DEPR_RATES = {
    "Maruti":0.10,"Hyundai":0.10,"Tata":0.09,"Honda":0.10,"Toyota":0.07,
    "Mahindra":0.12,"Ford":0.11,"Renault":0.11,"Skoda":0.10,"Volkswagen":0.10,
    "BMW":0.15,"Audi":0.15,"Mercedes-Benz":0.14,"Kia":0.10,"MG":0.11,
    "Jeep":0.10,"Nissan":0.11,"Datsun":0.12,"Chevrolet":0.12,"Fiat":0.13,
    "Mitsubishi":0.11,"Volvo":0.13,"Jaguar":0.15,"Land":0.13,"Lexus":0.13,
    "Daewoo":0.13,"Force":0.12,"Ambassador":0.14,"Ashok":0.12,"Isuzu":0.11,
    "Opel":0.13,"Peugeot":0.12,
}

@st.cache_resource
def load_model():
    cars = pd.read_csv('Car_details.csv')
    cars = cars.drop(columns=['torque'], errors='ignore')
    cars = cars.dropna().drop_duplicates().reset_index(drop=True)
    cars['name'] = cars['name'].apply(lambda x: str(x).split(' ')[0].strip())
    for col in ['mileage','engine','max_power']:
        cars[col] = [clean_unit(v) for v in cars[col]]
    cars = cars.dropna().reset_index(drop=True)
    cars['owner']        = cars['owner'].map(OWNER_MAP)
    cars['fuel']         = cars['fuel'].map(FUEL_MAP)
    cars['seller_type']  = cars['seller_type'].map(SELLER_MAP)
    cars['transmission'] = cars['transmission'].map(TRANSMISSION_MAP)
    cars['name']         = cars['name'].map(NAME_MAP)
    cars = cars.dropna().reset_index(drop=True)
    for col in ['mileage','engine','max_power','seats']:
        cars[col] = cars[col].astype(float)
    X = cars[['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']]
    y = cars['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mdl, round(r2*100,2), round(mae/100000,2)

@st.cache_data
def load_data():
    cars = pd.read_csv('Car_details.csv')
    cars['name'] = cars['name'].apply(lambda x: str(x).split(' ')[0].strip())
    return cars

model, r2, mae = load_model()
cars_data = load_data()
brands = sorted(cars_data['name'].unique())

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚗 AutoInsight Pro</h1>
  <p>ML-powered Car Price Prediction & Depreciation Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# ── MODEL STATS ────────────────────────────────────────────────────────────────
st.markdown("""<div class="section-card">
<div class="section-title">📊 ML Model Performance</div>
<div class="metric-row">""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-val green">{r2}%</div><div class="metric-lbl">Model Accuracy (R²)</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-val orange">₹{mae}L</div><div class="metric-lbl">Mean Avg Error</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="metric-val">8000+</div><div class="metric-lbl">Cars Trained On</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div class="metric-val">11</div><div class="metric-lbl">Features Used</div></div>', unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📉 Depreciation Analysis", "🔄 Compare Cars"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">🔮 Predict Car Selling Price</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        name         = st.selectbox('Car Brand', brands)
        year         = st.slider('Manufacturing Year', 1994, 2024, 2018)
        km_driven    = st.slider('Kms Driven', 11, 200000, 40000)
        fuel         = st.selectbox('Fuel Type', cars_data['fuel'].unique())
    with c2:
        seller_type  = st.selectbox('Seller Type', cars_data['seller_type'].unique())
        transmission = st.selectbox('Transmission', cars_data['transmission'].unique())
        owner        = st.selectbox('Owner Type', cars_data['owner'].unique())
        seats        = st.slider('Seats', 2, 14, 5)
    with c3:
        mileage      = st.slider('Mileage (kmpl)', 10, 42, 18)
        engine       = st.slider('Engine (CC)', 700, 3604, 1200)
        max_power    = st.slider('Max Power (bhp)', 0, 400, 85)

    if st.button("🔮 Predict Price", key="predict"):
        inp = pd.DataFrame([[
            NAME_MAP.get(name,1), year, km_driven,
            FUEL_MAP.get(fuel,1), SELLER_MAP.get(seller_type,1),
            TRANSMISSION_MAP.get(transmission,1), OWNER_MAP.get(owner,1),
            float(mileage), float(engine), float(max_power), float(seats)
        ]], columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
        price = model.predict(inp)[0]
        age = 2024 - year

        st.markdown(f"""
        <div class="result-banner">
            <div class="result-price">₹ {price:,.0f}</div>
            <div class="result-label">Estimated Market Value for {name} ({year})</div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2c, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-box"><div class="metric-val green">₹{price*0.85/100000:.1f}L</div><div class="metric-lbl">Min Est. (−15%)</div></div>', unsafe_allow_html=True)
        with r2c:
            st.markdown(f'<div class="metric-box"><div class="metric-val">₹{price/100000:.1f}L</div><div class="metric-lbl">Predicted Value</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-box"><div class="metric-val orange">₹{price*1.15/100000:.1f}L</div><div class="metric-lbl">Max Est. (+15%)</div></div>', unsafe_allow_html=True)

        insight = ""
        if age <= 3:
            insight = f"✅ This is a relatively new car ({age} years old). Resale value is strong."
        elif age <= 7:
            insight = f"ℹ️ This car is {age} years old — moderate depreciation expected."
        else:
            insight = f"⚠️ This car is {age} years old — significant depreciation has occurred."
        st.markdown(f'<div class="insight-box">{insight}<br>📌 Car has been driven {km_driven:,} kms with {transmission} transmission on {fuel} fuel.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — DEPRECIATION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">📉 Depreciation Analysis</div>', unsafe_allow_html=True)

    d1, d2 = st.columns([1, 2])
    with d1:
        dep_brand = st.selectbox("Car Brand", sorted(DEPR_RATES.keys()), key="dep_brand")
        dep_price = st.number_input("Original Price (₹)", min_value=100000, max_value=10000000, value=800000, step=50000)
        dep_years = st.slider("Years to Calculate", 1, 20, 10)

        if st.button("📊 Calculate", key="depr"):
            rate = DEPR_RATES[dep_brand]
            current_val = dep_price * ((1 - rate) ** dep_years)
            total_loss  = dep_price - current_val
            loss_pct    = (total_loss / dep_price) * 100
            insurance   = current_val * 0.035

            st.markdown(f"""
            <div class="result-banner" style="margin-top:1rem;">
                <div class="result-price">₹{current_val:,.0f}</div>
                <div class="result-label">Value after {dep_years} years</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="metric-box" style="margin:0.5rem 0"><div class="metric-val red">₹{total_loss:,.0f}</div><div class="metric-lbl">Total Value Lost</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box" style="margin:0.5rem 0"><div class="metric-val orange">{loss_pct:.1f}%</div><div class="metric-lbl">Depreciation %</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box" style="margin:0.5rem 0"><div class="metric-val green">₹{insurance:,.0f}/yr</div><div class="metric-lbl">Est. Insurance Premium</div></div>', unsafe_allow_html=True)

            if loss_pct > 60:
                st.markdown(f'<div class="warn-box">⚠️ High depreciation! {dep_brand} loses {int(rate*100)}% value per year. After {dep_years} years, {loss_pct:.0f}% value is lost.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box">✅ {dep_brand} has a relatively low depreciation rate of {int(rate*100)}%/year. Good resale value retention!</div>', unsafe_allow_html=True)

            st.session_state['dep_data'] = {
                'brand': dep_brand, 'price': dep_price, 'years': dep_years, 'rate': rate
            }

    with d2:
        if 'dep_data' in st.session_state:
            d = st.session_state['dep_data']
            years_range = list(range(0, d['years']+1))
            values = [d['price'] * ((1 - d['rate']) ** y) for y in years_range]

            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor('#0d0d1f')
            ax.set_facecolor('#0d0d1f')

            ax.fill_between(years_range, values, alpha=0.15, color='#7c5cfc')
            ax.plot(years_range, values, color='#7c5cfc', linewidth=2.5, marker='o', markersize=5)

            for i, (y, v) in enumerate(zip(years_range, values)):
                if i % 2 == 0 or i == len(years_range)-1:
                    ax.annotate(f'₹{v/100000:.1f}L', (y, v),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', fontsize=7.5, color='#aaa')

            ax.set_xlabel('Years', color='#888', fontsize=10)
            ax.set_ylabel('Car Value (₹)', color='#888', fontsize=10)
            ax.set_title(f'{d["brand"]} — Depreciation Over {d["years"]} Years', color='#e0d7ff', fontsize=12, fontweight='bold', pad=15)
            ax.tick_params(colors='#666')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/100000:.0f}L'))
            for spine in ax.spines.values():
                spine.set_edgecolor('#7c5cfc22')
            ax.grid(color='#7c5cfc15', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:300px;color:#555;font-size:1rem;">← Fill details and click Calculate</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — COMPARE CARS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">🔄 Compare Two Cars — Depreciation Head to Head</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown('<div class="compare-col"><div class="compare-title">🔵 Car A</div>', unsafe_allow_html=True)
        car_a_brand = st.selectbox("Brand", sorted(DEPR_RATES.keys()), key="ca_brand")
        car_a_price = st.number_input("Purchase Price (₹)", min_value=100000, max_value=10000000, value=700000, step=50000, key="ca_price")
        car_a_years = st.slider("Years", 1, 20, 5, key="ca_years")
        st.markdown('</div>', unsafe_allow_html=True)

    with cc2:
        st.markdown('<div class="compare-col"><div class="compare-title">🔴 Car B</div>', unsafe_allow_html=True)
        car_b_brand = st.selectbox("Brand", sorted(DEPR_RATES.keys()), index=4, key="cb_brand")
        car_b_price = st.number_input("Purchase Price (₹)", min_value=100000, max_value=10000000, value=700000, step=50000, key="cb_price")
        car_b_years = st.slider("Years", 1, 20, 5, key="cb_years")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Compare Now", key="compare"):
        max_years = max(car_a_years, car_b_years)
        yrs = list(range(0, max_years + 1))

        rate_a = DEPR_RATES[car_a_brand]
        rate_b = DEPR_RATES[car_b_brand]
        vals_a = [car_a_price * ((1 - rate_a) ** y) for y in range(0, car_a_years+1)]
        vals_b = [car_b_price * ((1 - rate_b) ** y) for y in range(0, car_b_years+1)]

        # Results
        r1, r2c = st.columns(2)
        final_a = vals_a[-1]
        final_b = vals_b[-1]
        loss_a  = car_a_price - final_a
        loss_b  = car_b_price - final_b

        with r1:
            st.markdown(f"""
            <div class="result-banner">
                <div style="color:#5c7cfc;font-size:0.9rem;font-weight:600;margin-bottom:4px">{car_a_brand}</div>
                <div class="result-price" style="color:#5c7cfc">₹{final_a:,.0f}</div>
                <div class="result-label">After {car_a_years} years | Lost: ₹{loss_a:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2c:
            st.markdown(f"""
            <div class="result-banner">
                <div style="color:#fc5c5c;font-size:0.9rem;font-weight:600;margin-bottom:4px">{car_b_brand}</div>
                <div class="result-price" style="color:#fc5c5c">₹{final_b:,.0f}</div>
                <div class="result-label">After {car_b_years} years | Lost: ₹{loss_b:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Chart
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0d0d1f')
        ax.set_facecolor('#0d0d1f')

        ax.fill_between(range(len(vals_a)), vals_a, alpha=0.12, color='#5c7cfc')
        ax.fill_between(range(len(vals_b)), vals_b, alpha=0.12, color='#fc5c5c')
        ax.plot(range(len(vals_a)), vals_a, color='#5c7cfc', linewidth=2.5, marker='o', markersize=5, label=car_a_brand)
        ax.plot(range(len(vals_b)), vals_b, color='#fc5c5c', linewidth=2.5, marker='s', markersize=5, label=car_b_brand)

        ax.set_xlabel('Years', color='#888', fontsize=10)
        ax.set_ylabel('Car Value (₹)', color='#888', fontsize=10)
        ax.set_title('Depreciation Comparison', color='#e0d7ff', fontsize=13, fontweight='bold', pad=15)
        ax.tick_params(colors='#666')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/100000:.0f}L'))
        for spine in ax.spines.values():
            spine.set_edgecolor('#7c5cfc22')
        ax.grid(color='#7c5cfc15', linestyle='--', linewidth=0.8)
        ax.legend(facecolor='#13132a', edgecolor='#7c5cfc33', labelcolor='#ddd', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Winner
        winner = car_a_brand if (final_a/car_a_price) > (final_b/car_b_price) else car_b_brand
        st.markdown(f'<div class="insight-box">🏆 <strong>{winner}</strong> retains more value proportionally! Better resale value = smarter buy.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;color:#444;font-size:0.8rem;">
    AutoInsight Pro — Built with Python, Scikit-learn & Streamlit &nbsp;|&nbsp; Developed by Dhruv Sikka
</div>
""", unsafe_allow_html=True)
