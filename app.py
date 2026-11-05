from flask import Flask, render_template, request, jsonify
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import mysql.connector

app = Flask(__name__)

import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="2587",
        database="lanka_fabric_shop",
        port="3305"
    )

# Load dataset
data = pd.read_csv("data/sales_data.csv")
data['sale_date'] = pd.to_datetime(data['sale_date'])
data['total_sales'] = data['quantity_sold'] * data['unit_price']

qty_model = joblib.load("models/sales_qty_model.pkl")
rev_model = joblib.load("models/revenue_model.pkl")
le = joblib.load("models/le_sku.pkl")

data = pd.read_csv("data/sales_data.csv")
data['sale_date'] = pd.to_datetime(data['sale_date'])
data['total_sales'] = data['quantity_sold'] * data['unit_price']

@app.route("/")
def dashboard():

    # Monthly sales
    data['month'] = data['sale_date'].dt.month
    monthly_sales = data.groupby('month')['total_sales'].sum()
    monthly_sales_list = [int(monthly_sales.get(i, 0)) for i in range(1, 13)]

    # Top products
    top_products = (
        data.groupby('product_name')['quantity_sold']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    # Top customers
    top_customers = (
        data.groupby('customer_name')['total_sales']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    return render_template(
        "dashboard.html",
        monthly_sales=monthly_sales_list,
        product_labels=list(top_products.index),
        product_values=[int(x) for x in top_products.values],
        customer_labels=list(top_customers.index),
        customer_values=[int(x) for x in top_customers.values],
        total_revenue=int(data['total_sales'].sum()),
        total_orders=len(data)
    )


@app.route("/predict_product", methods=["GET","POST"])
def predict_product():
    prediction = None
    if request.method == "POST":
        sku = request.form['sku']
        date = request.form['date']
        d = datetime.strptime(date,"%Y-%m-%d")
        sku_code = le.transform([sku])[0]
        prediction = int(qty_model.predict([[sku_code,d.month,d.day]])[0])
    return render_template("predict_product.html", prediction=prediction)

@app.route("/predict_revenue", methods=["GET","POST"])
def predict_revenue():
    revenue = None
    if request.method == "POST":
        year = int(request.form['year'])
        month = int(request.form['month'])
        revenue = int(rev_model.predict([[year,month]])[0])
    return render_template("predict_revenue.html", revenue=revenue)

@app.route("/top_products")
def top_products():

    df = data.copy()
    df['month'] = df['sale_date'].dt.month

    # Total quantity sold per product
    product_sales = df.groupby(['sku', 'product_name'])['quantity_sold'].sum().reset_index()

    # Average monthly sales (demand strength)
    monthly_avg = df.groupby(['sku','product_name','month'])['quantity_sold'].sum().groupby(level=[0,1]).mean().reset_index(name='avg_monthly_sales')

    # Merge both
    demand = pd.merge(product_sales, monthly_avg, on=['sku','product_name'])

    # Demand Score (Hybrid logic)
    demand['demand_score'] = demand['quantity_sold'] + demand['avg_monthly_sales']

    # Sort by demand score
    top_products = demand.sort_values(by='demand_score', ascending=False).head(5)

    return render_template("top_products.html", products=top_products)

@app.route("/top_customers")
def top_customers():

    df = data.copy()

    # Customer total spending
    customer_spend = df.groupby('customer_name').agg(
        total_spent=('total_sales', 'sum'),
        purchase_count=('quantity_sold', 'count'),
        avg_purchase=('total_sales', 'mean')
    ).reset_index()

    # Customer Score (Hybrid analytics)
    customer_spend['customer_score'] = (
        customer_spend['total_spent'] +
        (customer_spend['purchase_count'] * customer_spend['avg_purchase'])
    )

    # Sort top customers
    top_customers = customer_spend.sort_values(
        by='customer_score', ascending=False
    ).head(5)

    return render_template("top_customers.html", customers=top_customers)


@app.route("/model_accuracy")
def model_accuracy():

    model = joblib.load("models/sales_qty_model.pkl")
    le_sku = joblib.load("models/le_sku.pkl")

    df = data.copy()

    # SAME feature engineering as training
    df['day'] = df['sale_date'].dt.day
    df['month'] = df['sale_date'].dt.month
    df['sku_code'] = le_sku.transform(df['sku'])

    # ⚠️ SAME FEATURE ORDER AS TRAINING
    X = df[['sku_code', 'day', 'month']]
    y_actual = df['quantity_sold']

    y_pred = model.predict(X)

    # Accuracy metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    return render_template(
        "model_accuracy.html",
        rmse=round(rmse, 2),
        mae=round(mae, 2),
        r2=round(r2, 3)
    )

# -----------------------------
# FEATURE 9 – TOTAL QUANTITY PREDICTION FOR SKU PER MONTH
# -----------------------------
@app.route("/predict_sku_month", methods=["GET", "POST"])
def predict_sku_month():

    total_qty_prediction = None

    if request.method == "POST":
        sku = request.form['sku']
        year = int(request.form['year'])
        month = int(request.form['month'])

        model = joblib.load("models/sales_qty_model.pkl")
        le_sku = joblib.load("models/le_sku.pkl")

        sku_code = le_sku.transform([sku])[0]

        # Filter dates for the month
        days_in_month = pd.date_range(start=f"{year}-{month}-01",
                                      end=f"{year}-{month}-01") # We will iterate days dynamically
        # Get total days in that month
        total_days = pd.Period(f"{year}-{month}").days_in_month

        # Create feature matrix for all days
        X_list = [[sku_code, day, month] for day in range(1, total_days + 1)]
        X = pd.DataFrame(X_list, columns=['sku_code', 'day', 'month'])

        # Predict quantity per day
        y_pred = model.predict(X)

        # Sum total predicted quantity
        total_qty_prediction = int(sum(y_pred))

    return render_template("predict_sku_month.html", total_qty_prediction=total_qty_prediction)

# -----------------------------
# FEATURE 10 – TOTAL REVENUE PREDICTION FOR SKU OVER PERIOD
# -----------------------------
@app.route("/predict_sku_revenue", methods=["GET", "POST"])
def predict_sku_revenue():

    total_revenue = None

    if request.method == "POST":
        sku = request.form['sku']
        year = int(request.form['year'])
        start_month = int(request.form['start_month'])
        end_month = int(request.form['end_month'])

        qty_model = joblib.load("models/sales_qty_model.pkl")
        le_sku = joblib.load("models/le_sku.pkl")

        sku_code = le_sku.transform([sku])[0]

        total_days_list = []
        for month in range(start_month, end_month + 1):
            days_in_month = pd.Period(f"{year}-{month}").days_in_month
            total_days_list.extend([[sku_code, day, month] for day in range(1, days_in_month + 1)])

        X = pd.DataFrame(total_days_list, columns=['sku_code', 'day', 'month'])

        predicted_qty = qty_model.predict(X)

        # Average unit price of SKU
        avg_price = data[data['sku'] == sku]['unit_price'].mean()
        if pd.isna(avg_price):
            avg_price = 0

        total_revenue = int(sum(predicted_qty) * avg_price)

    return render_template("predict_sku_revenue.html", total_revenue=total_revenue)

@app.route("/predict_sku_date", methods=["GET", "POST"])
def predict_sku_date():

    predicted_qty = None

    if request.method == "POST":
        sku = request.form['sku']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        model = joblib.load("models/sales_qty_model.pkl")
        le_sku = joblib.load("models/le_sku.pkl")

        # Encode SKU
        sku_code = le_sku.transform([sku])[0]

        # Create feature input (must match training order!)
        X = pd.DataFrame([[sku_code, day, month]],
                         columns=['sku_code', 'day', 'month'])

        # Predict quantity
        predicted_qty = int(model.predict(X)[0])

    return render_template("predict_sku_date.html",
                           predicted_qty=predicted_qty)

# -----------------------------
# FEATURE 13 – CSV UPLOAD → NEXT 4 MONTHS SALES + DEMAND + PRODUCT QTY
# -----------------------------
@app.route("/upload_predict_4months", methods=["GET", "POST"])
def upload_predict_4months():

    predictions = None
    months = None
    top_products = None
    product_monthly_qty = None

    if request.method == "POST":

        file = request.files['csv_file']
        df = pd.read_csv(file)

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        # Total revenue column
        df['revenue'] = df['quantity'] * df['unit_price']

        # ---------- Train Revenue Model ----------
        monthly_sales = df.groupby(['year', 'month'])['revenue'].sum().reset_index()

        X = monthly_sales[['month']]
        y = monthly_sales['revenue']

        model = LinearRegression()
        model.fit(X, y)

        # ---------- Predict Next 4 Months ----------
        last_month = monthly_sales['month'].max()
        next_months = [(last_month + i - 1) % 12 + 1 for i in range(1, 5)]

        preds = model.predict(pd.DataFrame(next_months, columns=['month']))
        predictions = [int(p) for p in preds]
        months = [f"Month +{i}" for i in range(1, 5)]

        # ---------- Top Demand Products ----------
        product_totals = df.groupby('product_name')['quantity'].sum().reset_index()
        top_products = product_totals.sort_values(by='quantity', ascending=False).head(5)

        # ---------- Monthly Quantity Per Product ----------
        product_monthly = df.groupby(['product_name', 'month'])['quantity'].sum().reset_index()

        product_monthly_qty = product_monthly.to_dict(orient="records")

    return render_template(
        "upload_predict_4months.html",
        predictions=predictions,
        months=months,
        top_products=top_products,
        product_monthly_qty=product_monthly_qty
    )

# -----------------------------
# FEATURE 12 – NEXT 3 MONTHS BUSINESS SALES PREDICTION
# -----------------------------
@app.route("/predict_next_3_months", methods=["GET", "POST"])
def predict_next_3_months():

    predictions = None
    months = None

    if request.method == "POST":
        current_sales = float(request.form['current_sales'])

        # Simple growth logic (can be replaced with ML later)
        growth_rate = 0.08 # 8% monthly growth

        month1 = int(current_sales * (1 + growth_rate))
        month2 = int(month1 * (1 + growth_rate))
        month3 = int(month2 * (1 + growth_rate))

        predictions = [month1, month2, month3]
        months = ["Next Month", "Month +2", "Month +3"]

    return render_template(
        "predict_next_3_months.html",
        predictions=predictions,
        months=months
    )

@app.route("/master_forecast", methods=["GET", "POST"])
def master_forecast():
    results = []
    charts_data = []
    revenue_forecast = []
    top_products = []

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file:
            return render_template("master_forecast.html",
                                   error="Please upload a CSV file")

        df = pd.read_csv(file)

        required_cols = ["sku", "product_name", "quantity", "unit_price"]
        if not all(col in df.columns for col in required_cols):
            return render_template("master_forecast.html",
                                   error=f"CSV must contain: {required_cols}")

        # Handle date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
        elif "year" in df.columns and "month" in df.columns:
            df["year"] = df["year"].astype(int)
            df["month"] = df["month"].astype(int)
        else:
            return render_template("master_forecast.html",
                                   error="CSV must have 'date' or 'year' + 'month'")

        # ======================
        # BUSINESS REVENUE MODEL
        # ======================
        df["revenue"] = df["quantity"] * df["unit_price"]
        monthly_rev = df.groupby(["year", "month"])["revenue"].sum().reset_index()

        X_rev = monthly_rev[["month"]]
        y_rev = monthly_rev["revenue"]
        rev_model = LinearRegression()
        rev_model.fit(X_rev, y_rev)

        last_month = monthly_rev["month"].max()
        next_months = [(last_month + i - 1) % 12 + 1 for i in range(1, 5)]
        next_years = [monthly_rev["year"].max()
                      if m > last_month else monthly_rev["year"].max() + 1
                      for m in next_months]

        rev_preds = rev_model.predict(pd.DataFrame({"month": next_months}))
        rev_preds = [round(max(0, r), 2) for r in rev_preds]

        for i in range(4):
            revenue_forecast.append({
    "year": int(next_years[i]),
    "month": int(next_months[i]),
    "revenue": float(rev_preds[i])
})

        # ======================
        # PER SKU FORECAST + DYNAMIC PRICE
        # ======================
        all_skus = df["sku"].unique()

        for sku in all_skus:
            sku_df = df[df["sku"] == sku]
            monthly = sku_df.groupby(["year", "month"]).agg({
                "quantity": "sum",
                "unit_price": "mean"
            }).reset_index()

            X = monthly[["month"]]
            y = monthly["quantity"]
            model = LinearRegression()
            model.fit(X, y)

            next_months_sku = [(last_month + i - 1) % 12 + 1 for i in range(1, 4)]
            next_years_sku = [monthly["year"].max()
                              if m > last_month else monthly["year"].max() + 1
                              for m in next_months_sku]

            preds_qty = model.predict(pd.DataFrame({"month": next_months_sku}))
            preds_qty = [int(max(0, p)) for p in preds_qty]

            avg_qty = monthly["quantity"].mean()
            last_price = monthly["unit_price"].iloc[-1]

            sku_result = []

            for i in range(3):
                if preds_qty[i] > avg_qty:
                    new_price = last_price * 1.08
                    label = "High Demand (+8%)"
                else:
                    new_price = last_price * 0.95
                    label = "Low Demand (-5%)"

                sku_result.append({
                    "sku": sku,
                    "product_name": sku_df["product_name"].iloc[0],
                    "year": next_years_sku[i],
                    "month": next_months_sku[i],
                    "predicted_quantity": preds_qty[i],
                    "last_price": round(last_price, 2),
                    "suggested_price": round(new_price, 2),
                    "demand_label": label
                })

            results.append(sku_result)

            charts_data.append({
                "sku": sku,
                "product_name": sku_df["product_name"].iloc[0],
                "months": [f"{next_years_sku[i]}-{next_months_sku[i]}" for i in range(3)],
                "quantities": [int(q) for q in preds_qty],
"prices": [float(r["suggested_price"]) for r in sku_result]
            })

        # ======================
        # TOP PRODUCTS
        # ======================
        top_products_df = df.groupby(["sku", "product_name"])["quantity"] \
                         .sum().reset_index() \
                         .sort_values(by="quantity", ascending=False).head(5)
        top_products_df["quantity"] = top_products_df["quantity"].astype(int)
        top_products = top_products_df.to_dict(orient="records")
        
    return render_template("master_forecast.html",
                           results=results,
                           charts_data=charts_data,
                           revenue_forecast=revenue_forecast,
                           top_products=top_products,
                           error=None)

if __name__ == "__main__":
    app.run(debug=True)
