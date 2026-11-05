import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# ✅ Ensure models folder exists
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/sales_data.csv")

df['sale_date'] = pd.to_datetime(df['sale_date'])
df['month'] = df['sale_date'].dt.month
df['day'] = df['sale_date'].dt.day

le = LabelEncoder()
df['sku_code'] = le.fit_transform(df['sku'])

X = df[['sku_code', 'day', 'month']]
y = df['quantity_sold']

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/sales_qty_model.pkl")
joblib.dump(le, "models/le_sku.pkl")

print("✅ Product Quantity Model Trained Successfully")

