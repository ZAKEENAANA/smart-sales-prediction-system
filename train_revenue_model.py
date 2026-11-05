import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("data/sales_data.csv")

df['sale_date'] = pd.to_datetime(df['sale_date'])
df['month'] = df['sale_date'].dt.month
df['year'] = df['sale_date'].dt.year
df['total_sales'] = df['quantity_sold'] * df['unit_price']

monthly = df.groupby(['year','month'])['total_sales'].sum().reset_index()

X = monthly[['year','month']]
y = monthly['total_sales']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/revenue_model.pkl")

print("✅ Revenue Prediction Model Trained Successfully")
